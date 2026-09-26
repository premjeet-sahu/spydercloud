import os
import unittest
from unittest.mock import MagicMock, patch

from botocore.exceptions import ProfileNotFound

from jev_poc.providers import BedrockProvider
from jev_poc.runtime import credential_hint, sanitize


class CredentialResolutionTests(unittest.TestCase):
    def test_access_keys_take_precedence_over_invalid_profile(self):
        environment = {
            "AWS_ACCESS_KEY_ID": "AKIATEST1234",
            "AWS_SECRET_ACCESS_KEY": "not-a-real-secret",
            "AWS_PROFILE": "profile-that-does-not-exist",
            "AWS_REGION": "us-east-1",
        }
        session = MagicMock()
        with patch.dict(os.environ, environment, clear=True), patch("boto3.Session", return_value=session) as factory:
            provider = BedrockProvider()

        self.assertEqual("environment_access_keys", provider.credential_source)
        self.assertEqual("AKIATEST1234", factory.call_args.kwargs["aws_access_key_id"])
        self.assertNotIn("profile_name", factory.call_args.kwargs)
        session.client.assert_called_once()

    def test_missing_profile_falls_back_to_default_chain(self):
        fallback_session = MagicMock()
        with patch.dict(os.environ, {"AWS_PROFILE": "missing", "AWS_REGION": "us-east-1"}, clear=True), patch(
            "boto3.Session",
            side_effect=[ProfileNotFound(profile="missing"), fallback_session],
        ) as factory:
            provider = BedrockProvider()

        self.assertEqual("default_chain_after_missing_profile", provider.credential_source)
        self.assertEqual(2, factory.call_count)
        self.assertNotIn("profile_name", factory.call_args.kwargs)

    def test_partial_access_keys_are_rejected(self):
        with patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": "AKIATEST1234"}, clear=True):
            with self.assertRaisesRegex(Exception, "must both be set"):
                BedrockProvider()

    def test_diagnostics_mask_access_key_and_secrets(self):
        environment = {
            "AWS_ACCESS_KEY_ID": "AKIATEST1234",
            "AWS_SECRET_ACCESS_KEY": "secret",
        }
        with patch.dict(os.environ, environment, clear=True):
            hint = credential_hint()
        self.assertEqual("***1234", hint["access_key"])
        self.assertEqual("[REDACTED]", sanitize({"authorization": "Bearer secret"})["authorization"])


if __name__ == "__main__":
    unittest.main()
