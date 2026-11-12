"""Workspace resolution uses read-only discovery unless a team name is set."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from starVLA.training.trainer_utils import wandb_setup


class WandbSetupTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"WANDB_MODE": "online", "WANDB_ENTITY": "", "WANDB_CREATE_TEAM": ""})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def api(self, *, default=None, teams=(), username=None, access=None, project_read_only=None):
        api = MagicMock()
        api.default_entity = default
        api.viewer = SimpleNamespace(entity=default, teams=list(teams), username=username)
        def permissions(_query, variables):
            read_only = (access or {}).get(variables["entity"], False)
            return {"entity": {"name": variables["entity"], "readOnly": read_only},
                    "project": None if project_read_only is None else {"readOnly": project_read_only}}
        api._service_api.execute_graphql.side_effect = permissions
        return api

    def test_explicit_workspace_is_checked_without_creation(self):
        api = self.api()
        with patch.dict(os.environ, {"WANDB_ENTITY": "my-team"}), patch.object(wandb_setup.wandb, "Api", return_value=api):
            self.assertEqual(wandb_setup.resolve_wandb_entity({}), "my-team")
        self.assertEqual(api._service_api.execute_graphql.call_args.args[1]["entity"], "my-team")
        api.create_team.assert_not_called()

    def test_api_default_is_passed_explicitly_to_wandb(self):
        api = self.api(default="my-default")
        with patch.object(wandb_setup.wandb, "Api", return_value=api):
            self.assertEqual(wandb_setup.resolve_wandb_entity({}), "my-default")
        api.create_team.assert_not_called()

    def test_unique_existing_team_is_selected_without_creation(self):
        api = self.api(teams=["robotics"])
        with patch.object(wandb_setup.wandb, "Api", return_value=api):
            self.assertEqual(wandb_setup.resolve_wandb_entity({}), "robotics")
        api.create_team.assert_not_called()

    def test_ambiguous_or_missing_workspaces_do_not_create_anything(self):
        for teams in ([], ["lab-a", "lab-b"]):
            api = self.api(teams=teams)
            with patch.object(wandb_setup.wandb, "Api", return_value=api), self.assertRaisesRegex(ValueError, "TRAIN_WANDB"):
                wandb_setup.resolve_wandb_entity({})
            api.create_team.assert_not_called()

    def test_only_an_explicit_new_name_triggers_creation(self):
        api = self.api(default="other-team")
        with patch.dict(os.environ, {"WANDB_CREATE_TEAM": "vlact-new-team"}), \
                patch.object(wandb_setup.wandb, "Api", return_value=api):
            self.assertEqual(wandb_setup.resolve_wandb_entity({}), "vlact-new-team")
        api.create_team.assert_called_once_with("vlact-new-team")

    def test_named_existing_team_is_reused(self):
        api = self.api(teams=["vlact-existing"])
        with patch.dict(os.environ, {"WANDB_CREATE_TEAM": "vlact-existing"}), \
                patch.object(wandb_setup.wandb, "Api", return_value=api):
            self.assertEqual(wandb_setup.resolve_wandb_entity({}), "vlact-existing")
        api.create_team.assert_not_called()

    def test_offline_default_does_not_query_account(self):
        with patch.dict(os.environ, {"WANDB_MODE": "offline"}), patch.object(wandb_setup.wandb, "Api") as factory:
            self.assertIsNone(wandb_setup.resolve_wandb_entity({}))
            factory.assert_not_called()

    def test_check_only_never_creates_a_team(self):
        api = self.api()
        with patch.dict(os.environ, {"WANDB_CREATE_TEAM": "new-team"}), \
                patch.object(wandb_setup.wandb, "Api", return_value=api), self.assertRaisesRegex(ValueError, "check-only"):
            wandb_setup.resolve_wandb_entity({}, allow_create=False)
        api.create_team.assert_not_called()

    def test_read_only_default_never_falls_back_to_a_login_username(self):
        api = self.api(default="models", username="personal-account", access={"models": True})
        with patch.object(wandb_setup.wandb, "Api", return_value=api), self.assertRaisesRegex(ValueError, "No writable"):
            wandb_setup.resolve_wandb_entity({})
        attempted = [call.args[1]["entity"] for call in api._service_api.execute_graphql.call_args_list]
        self.assertEqual(attempted, ["models"])
        api.create_team.assert_not_called()

    def test_explicit_read_only_or_unknown_entity_is_rejected(self):
        for read_only in (True, None):
            api = self.api(access={"chosen-team": read_only})
            with patch.dict(os.environ, {"WANDB_ENTITY": "chosen-team"}), \
                    patch.object(wandb_setup.wandb, "Api", return_value=api), self.assertRaises(PermissionError):
                wandb_setup.resolve_wandb_entity({})

    def test_existing_read_only_project_is_rejected(self):
        api = self.api(default="my-team", project_read_only=True)
        with patch.object(wandb_setup.wandb, "Api", return_value=api), self.assertRaisesRegex(ValueError, "No writable"):
            wandb_setup.resolve_wandb_entity({"wandb_project": "private-project"})

    def test_failed_team_creation_is_not_treated_as_writable(self):
        api = self.api(access={"new-team": True})
        with patch.dict(os.environ, {"WANDB_CREATE_TEAM": "new-team"}), \
                patch.object(wandb_setup.wandb, "Api", return_value=api), self.assertRaises(PermissionError):
            wandb_setup.resolve_wandb_entity({})
