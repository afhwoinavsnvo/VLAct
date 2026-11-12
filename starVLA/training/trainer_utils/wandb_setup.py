"""Resolve a W&B workspace, with explicitly named team creation when requested."""

import argparse
import os

from omegaconf import OmegaConf
import wandb


def validate_wandb_access(api, entity, project):
    """Check minimal permission metadata without creating any W&B resources."""
    data = api._service_api.execute_graphql(
        """query VLActWandbAccess($entity: String!, $project: String!) {
          entity(name: $entity) { name readOnly }
          project(entityName: $entity, name: $project) { readOnly }
        }""",
        {"entity": entity, "project": project},
    )
    workspace = data.get("entity")
    if not workspace or workspace.get("readOnly") is not False:
        raise PermissionError(f"W&B entity '{entity}' is missing, inaccessible or read-only for this API key")
    existing_project = data.get("project")
    if existing_project is not None and existing_project.get("readOnly") is not False:
        raise PermissionError(f"W&B project '{entity}/{project}' is read-only for this API key")
    # A null project can mean a new project (created by wandb.init) or an
    # inaccessible existing project. Entity write access is verified here;
    # the final run initialization remains authoritative for project creation.
    return entity


def resolve_wandb_entity(config, *, allow_create=True):
    entity = os.environ.get("WANDB_ENTITY") or config.get("wandb_entity")
    if entity in ("your_name", "your_wandb_entity"):
        entity = None
    create_name = os.environ.get("WANDB_CREATE_TEAM") or config.get("wandb_create_team")
    if entity and create_name and entity != create_name:
        raise ValueError("Choose one W&B workspace: WANDB_ENTITY and WANDB_CREATE_TEAM disagree")
    mode = os.environ.get("WANDB_MODE", "online")
    if mode != "online":
        if create_name:
            raise ValueError("Creating a W&B team requires online mode")
        return entity

    api = wandb.Api(timeout=20)
    project = os.environ.get("WANDB_PROJECT") or config.get("wandb_project") or "uncategorized"
    if entity:
        return validate_wandb_access(api, entity, project)
    viewer = api.viewer
    teams = sorted(set(viewer.teams))
    if create_name:
        if create_name == viewer.entity or create_name in teams:
            return validate_wandb_access(api, create_name, project)
        if not allow_create:
            raise ValueError(f"Team '{create_name}' is not an existing workspace; check-only does not create teams")
        # create_team uses the account's Default API organization. Permissions,
        # organization selection and name conflicts are enforced by W&B.
        api.create_team(create_name)
        return validate_wandb_access(api, create_name, project)

    rejected = []
    # Login usernames are not necessarily run-capable entities. In particular,
    # read-only entity metadata cannot establish that a username can own runs.
    candidates = [api.default_entity]
    checked = set()
    for candidate in candidates:
        if not candidate or candidate in checked:
            continue
        checked.add(candidate)
        try:
            return validate_wandb_access(api, candidate, project)
        except PermissionError as error:
            rejected.append(str(error))
    writable_teams = []
    for team in teams:
        if team in checked:
            continue
        try:
            writable_teams.append(validate_wandb_access(api, team, project))
        except PermissionError as error:
            rejected.append(str(error))
    if len(writable_teams) == 1:
        return writable_teams[0]
    if writable_teams:
        raise ValueError(f"Multiple writable W&B workspaces are available: {', '.join(writable_teams)}. Set TRAIN_WANDB_ENTITY.")
    raise ValueError(
        "No writable W&B entity is available. Set TRAIN_WANDB_ENTITY to your workspace, "
        "or TRAIN_WANDB_CREATE_TEAM to a new team name (requires a Default API organization and creation permissions). "
        + " ".join(rejected)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--check-only", action="store_true", help="Never create a team")
    args = parser.parse_args()
    entity = resolve_wandb_entity(OmegaConf.load(args.config), allow_create=not args.check_only)
    if not entity:
        parser.error("No online W&B workspace was selected")
    print(entity)


if __name__ == "__main__":
    main()
