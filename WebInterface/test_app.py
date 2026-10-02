"""Run: python -m unittest WebInterface.test_app (no API calls)."""
import io
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import time
import unittest
from unittest.mock import patch, Mock

from fastapi.testclient import TestClient
import numpy as np

from WebInterface import app as server
from WebInterface.geometry import read_lattice
from utils.supervisor_response import parse_supervisor_response
from WebInterface.runtime import log_exception
from WebInterface.worker import bounded_responses
from deploy.cleanup_public_runs import expired_paths


def lattice_bytes(**changes):
    fields = dict(frac_coords=np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                  edge_index=np.array([[0, 1, 0, 1], [1, 0, 2, 2]]),
                  lengths=np.array([2., 3, 4]), angles=np.array([90., 90, 90]),
                  prop_list=np.array(None, dtype=object))
    fields.update(changes)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fields)
    return buffer.getvalue()


class WorkbenchChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.results = root / "results"
        self.results.mkdir()
        (self.results / "sample.npz").write_bytes(lattice_bytes())
        self.patches = [patch.object(server, "RESULTS", self.results),
                        patch.object(server, "LOCAL", root / "local"),
                        patch.dict("os.environ", {"OPENAI_API_KEY": "test-credential-not-a-real-key"})]
        for item in self.patches:
            item.start()
        server.CACHE.clear()
        server.READ_LIMITS.clear()
        self.client = TestClient(server.app).__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def test_geometry_preserves_scale_and_includes_isolated_nodes(self):
        geometry = read_lattice(lattice_bytes())
        np.testing.assert_allclose(geometry["cart_coords"], [[0,0,0],[2,0,0],[0,3,0],[0,0,4]], atol=1e-12)
        self.assertEqual(geometry["components"], 2)
        self.assertEqual(geometry["struts"], 3)
        self.assertEqual(geometry["stored_edges"], 4)
        self.assertIsNone(geometry["predictions"])

    def test_skew_cell_preserves_lengths_and_angles(self):
        geometry = read_lattice(lattice_bytes(angles=np.array([80.,75,65])))
        basis = np.array(geometry["basis"])
        np.testing.assert_allclose(np.linalg.norm(basis, axis=1), [2,3,4])
        for (a,b), angle in zip([(1,2),(0,2),(0,1)], [80,75,65]):
            self.assertAlmostEqual(float(np.degrees(np.arccos(np.dot(basis[a],basis[b]) / (np.linalg.norm(basis[a])*np.linalg.norm(basis[b]))))), angle)

    def test_rejects_invalid_geometry_and_preserves_prediction_sign(self):
        invalid = [dict(frac_coords=np.array([[np.nan,0,0]])),
                   dict(edge_index=np.array([[0],[50]])),
                   dict(edge_index=np.array([[0.],[1.]])),
                   dict(lengths=np.array([-1.,1,1])),
                   dict(angles=np.array([5.,5,175])),
                   dict(frac_coords=np.array([["bad",0,0]],dtype=object))]
        for changes in invalid:
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                read_lattice(lattice_bytes(**changes))
        prediction = [1e-5] * 6 + [-.3] * 6
        self.assertEqual(read_lattice(lattice_bytes(y_pred=prediction))["predictions"], prediction)

    def test_import_roundtrip_and_original_download(self):
        content = lattice_bytes()
        response = self.client.post("/api/import", content=content)
        self.assertEqual(response.status_code, 201)
        identifier = response.json()["id"]
        self.assertEqual(self.client.get(f"/api/candidates/{identifier}/download").content, content)
        self.assertEqual(self.client.get(f"/api/candidates/{identifier}").json()["evidence"], "Geometry only")
        self.assertEqual(self.client.post("/api/import", content=b"not a zip").status_code, 422)

    def test_private_files_and_cross_site_jobs_are_inaccessible(self):
        for path in ["/.env", "/WebInterface/.env", "/../.env", "/etc/passwd", "/.env.local", "/static/../.env", "/static/%2e%2e/.env", "/WebInterface/.env.local", "/api/candidates/..%2F.env/download", "/api/runs/..%2F.env", "/api/runs/"+"z"*32, "/checkpoints/best_ae_model.pt", "/openapi.json"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 404, path)
        self.assertEqual(self.client.post("/api/runs", headers={"Origin":"https://attacker.example"}, json={"kind":"prediction"}).status_code, 403)
        self.assertEqual(self.client.get("/api/status", headers={"Host":"attacker.example"}).status_code, 403)
        self.assertNotIn("test-credential-not-a-real-key", self.client.get("/api/status").text)

    def test_health_and_payload_limits(self):
        response = self.client.get("/api/health")
        self.assertEqual(set(response.json()), {"status", "web", "generation_available", "prediction_available"})
        for payload in [dict(prompt="x" * 2001), dict(prompt="x", attempts=3),
                        dict(prompt="x", attempts=True), dict(prompt="x", source_path="/etc/passwd"),
                        dict(prompt="x", candidate_id="../.env"), dict(prompt="x", logic_mode="; echo bad"), dict(prompt="x", condition=[1e308]*12)]:
            self.assertEqual(self.client.post("/api/runs", json={"kind":"generation", **payload}).status_code, 422)
        self.assertEqual(self.client.post("/api/runs", content=b"x" * 16385).status_code, 413)
        self.assertEqual(self.client.post("/api/import", content=b"x" * (4*1024*1024+1)).status_code, 413)
        self.assertEqual(self.client.post("/api/runs", headers={"Sec-Fetch-Site":"cross-site"}, json={"kind":"prediction"}).status_code, 403)

    def test_symlinks_cannot_escape_catalog(self):
        sample = self.results / "sample.npz"
        identifier = self.client.get("/api/candidates").json()["items"][0]["id"]
        outside = self.results.parent / "private.npz"
        sample.rename(outside)
        sample.symlink_to(outside)
        self.assertEqual(self.client.get(f"/api/candidates/{identifier}/download").status_code, 404)
        server.index_files()
        self.assertNotIn(identifier, server.FILES)

    def test_public_sessions_and_persistent_generation_rate_limit(self):
        with patch.object(server, "PUBLIC", True), patch.object(server, "capabilities", return_value={"prediction":True,"generation":True}), patch.object(server, "run_worker"):
            self.client.get("/api/status")
            imported = self.client.post("/api/import", content=lattice_bytes()).json()["id"]
            other = TestClient(server.app)
            other.portal = self.client.portal
            other.get("/api/status")
            self.assertEqual(other.get(f"/api/candidates/{imported}").status_code, 404)
            self.assertEqual(other.get(f"/api/candidates/{imported}/download").status_code, 404)
            self.assertNotIn(imported, [x["id"] for x in other.get("/api/candidates").json()["items"]])
            for _ in range(2):
                response = self.client.post("/api/runs", json={"kind":"generation", "prompt":"simple truss"})
                self.assertEqual(response.status_code, 202)
                identifier = response.json()["id"]
                self.assertEqual(other.get("/api/runs/" + identifier).status_code, 404)
                self.assertEqual(other.get("/api/runs").json(), [])
                self.assertNotIn("owner", self.client.get("/api/runs/" + identifier).json())
                self.assertEqual(other.post("/api/runs", json={"kind":"generation", "prompt":"simple truss"}).status_code, 409)
                path = server.LOCAL / "runs" / identifier / "status.json"
                item = json.loads(path.read_text())
                item["state"] = "complete"
                server.write_json(path, item)
            server.READ_LIMITS.clear()  # Simulate clearing process-local state on restart.
            response = other.post("/api/runs", headers={"X-Forwarded-For":"8.8.8.8"}, json={"kind":"generation", "prompt":"simple truss"})
            self.assertEqual(response.status_code, 429)
            self.assertIn("Retry-After", response.headers)
            self.assertTrue((server.LOCAL / "limits.json").is_file())

    def test_timeout_kills_worker_and_preserves_failure(self):
        with patch.object(server, "capabilities", return_value={"prediction":True,"generation":True}), patch.object(server, "run_worker"):
            identifier = self.client.post("/api/runs", json={"kind":"generation", "prompt":"simple truss"}).json()["id"]
        with patch.object(server.subprocess, "Popen") as process:
            process.return_value.wait.side_effect = [subprocess.TimeoutExpired("worker", 900), 0]
            process.return_value.poll.return_value = None
            server.run_worker(server.LOCAL / "runs" / identifier)
            process.return_value.kill.assert_called_once()
        status = self.client.get("/api/runs/" + identifier).json()
        self.assertEqual(status["state"], "failed")
        self.assertEqual(status["stage"], "Run timed out")

    def test_public_read_limit_and_global_spending_cap(self):
        with patch.object(server, "PUBLIC", True), patch.object(server, "capabilities", return_value={"prediction":True,"generation":True}):
            server.READ_LIMITS.update(minute=int(time.time()//60), testclient=120)
            self.assertEqual(self.client.get("/api/candidates").status_code, 429)
            self.assertEqual(self.client.get("/api/health").status_code, 200)
            server.READ_LIMITS.clear()
            server.write_json(server.LOCAL / "limits.json", [dict(action="generation", client="other", time=time.time()) for _ in range(24)])
            response = self.client.post("/api/runs", json={"kind":"generation", "prompt":"simple truss"})
            self.assertEqual(response.status_code, 429)
            self.assertEqual(self.client.get("/api/runs").json(), [])

    def test_worker_candidate_is_available_before_worker_exit(self):
        name = "prediction-" + "a" * 32 + "-1.npz"
        (server.LOCAL / "candidates" / name).write_bytes(lattice_bytes(y_pred=[.1]*12))
        identifier = hashlib.sha256(f"workspace/{name}".encode()).hexdigest()[:24]
        self.assertNotIn(identifier, server.FILES)
        response = self.client.get("/api/candidates/" + identifier)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["evidence"], "Predicted")

    def test_provider_budget_and_secret_free_diagnostics(self):
        client = Mock()
        client.with_options.return_value = client
        provider_call = client.responses.create
        guard = Mock(side_effect=[None, RuntimeError("budget exhausted")])
        bounded = bounded_responses(client, guard, "Supervisor evaluation")
        bounded.responses.create(model="example", input="a truss")
        client.with_options.assert_called_once_with(timeout=60, max_retries=0)
        provider_call.assert_called_once_with(model="example", input="a truss", max_output_tokens=2500)
        with self.assertRaises(RuntimeError):
            bounded.responses.create(model="example", input="a truss")
        self.assertEqual(provider_call.call_count, 1)
        try:
            raise ValueError("test-credential-not-a-real-key")
        except ValueError as exc:
            with self.assertLogs(server.LOGGER, level="ERROR") as capture:
                log_exception(server.LOGGER, "example-run", exc)
        self.assertNotIn("test-credential-not-a-real-key", str(capture.output))
        self.assertIn("ValueError", str(capture.output))
        with patch.object(server, "candidate", side_effect=RuntimeError("test-credential-not-a-real-key")), self.assertLogs(server.LOGGER, level="ERROR") as capture:
            response = self.client.get("/api/candidates/"+"a"*24)
        self.assertEqual(response.status_code,500)
        self.assertNotIn("test-credential-not-a-real-key",response.text+str(capture.output))

    def test_cleanup_excludes_active_jobs_and_research_results(self):
        old = time.time() - 10 * 86400
        for identifier, state in [("a"*32,"complete"),("b"*32,"running")]:
            server.write_json(server.LOCAL / "runs" / identifier / "status.json",dict(state=state,updated=old))
            name = "generation-" + identifier + "-1"
            server.write_json(server.LOCAL / "candidates" / (name+".json"),dict(run_id=identifier,created=old))
            (server.LOCAL / "candidates" / (name+".npz")).write_bytes(lattice_bytes())
        paths = expired_paths(server.LOCAL,7)
        self.assertEqual(len(paths),3)
        self.assertTrue(all("b"*32 not in str(path) and path.is_relative_to(server.LOCAL) for path in paths))
        self.assertTrue((self.results / "sample.npz").is_file())

    def test_run_validation_and_duplicate_submission(self):
        identifier = self.client.get("/api/candidates").json()["items"][0]["id"]
        with patch.object(server, "capabilities", return_value={"prediction":True,"generation":True}), patch.object(server, "run_worker"):
            self.assertEqual(self.client.post("/api/runs", json={"kind":"generation","prompt":"","attempts":0}).status_code, 422)
            self.assertEqual(self.client.post("/api/runs", json={"kind":"generation","prompt":"x","condition":[1,2]}).status_code, 422)
            response = self.client.post("/api/runs", json={"kind":"prediction","candidate_id":identifier})
            self.assertEqual(response.status_code, 202)
            self.assertEqual(self.client.post("/api/runs", json={"kind":"prediction","candidate_id":identifier}).status_code, 409)
            run = self.client.get("/api/runs/" + response.json()["id"]).json()
            self.assertEqual(run["state"], "queued")
            self.assertNotIn("test-credential-not-a-real-key", json.dumps(run))

    def test_supervisor_signed_exponents_and_malformed_output(self):
        text = """Score: 0.7
Improved Prompt: Keep a negative Poisson response.
Improved Properties:
Young's modulus: [1e-4, 2E-3, .003]
Shear modulus: [1e-5, 2e-5, 3e-5]
Poisson ratio: [-.3, -.2, -.1, .1, .2, .3]"""
        score, prompt, properties = parse_supervisor_response(text)
        self.assertEqual(score, .7)
        self.assertEqual(properties[0], 1e-4)
        self.assertEqual(properties[6], -.3)
        self.assertEqual(len(properties), 12)
        with self.assertRaises(ValueError):
            parse_supervisor_response(text.replace("Score: 0.7", "Score: 2"))
        with self.assertRaises(ValueError):
            parse_supervisor_response(text.replace("[1e-4, 2E-3, .003]", "[1, 2]"))


if __name__ == "__main__":
    unittest.main()
