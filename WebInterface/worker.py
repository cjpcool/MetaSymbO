"""Isolated model worker; credentials stay in the environment, never run files."""
import hashlib
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

from .geometry import read_lattice
from .runtime import write_json, configure_logging, log_exception

ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("metasymbo.worker")


def bounded_responses(client, count_call, stage):
    """Bound even legacy parsing retries without changing the scientific model."""
    client = client.with_options(timeout=60, max_retries=0)
    original = client.responses.create

    def create(**kwargs):
        count_call(stage)
        if len(str(kwargs.get("input", ""))) > 30000:
            raise ValueError("Generated review exceeded the input limit")
        return original(**kwargs, max_output_tokens=2500)

    client.responses.create = create
    return client


def main(directory):
    request = json.loads((directory / "request.json").read_text())
    status = json.loads((directory / "status.json").read_text())
    output = directory.parents[1] / "candidates"
    checkpoint = Path(os.environ["METASYMBO_CHECKPOINT"])

    def update(stage, **fields):
        status.update(stage=stage, updated=time.time(), **fields)
        write_json(directory / "status.json", status)
        LOGGER.info("run=%s state=%s stage=%s duration=%.1fs", status["id"], status["state"], stage, time.time() - status["created"])

    def save_candidate(coords, edges, lengths, angles, predictions, **provenance):
        name = f"{request['kind']}-{status['id']}-{len(status['candidates']) + 1}"
        path = output / f"{name}.npz"
        def array(value):
            return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
        values = {"frac_coords": array(coords), "edge_index": array(edges),
                  "lengths": array(lengths).reshape(-1), "angles": array(angles).reshape(-1),
                  "y_pred": array(predictions).reshape(-1)}
        values["atom_types"] = np.ones(len(values["frac_coords"]), dtype=np.int64)
        if request.get("condition") is not None:
            values["condition"] = np.asarray(request["condition"])
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **values)
        read_lattice(buffer.getvalue())
        write_json(path.with_suffix(".json"), dict(run_id=status["id"], kind=request["kind"],
                   owner=request.get("owner", "local"),
                   parent_id=request.get("candidate_id"), prompt=request["prompt"], seed=request["seed"],
                   checkpoint=checkpoint.name, property_units="Model scale; physical units unverified",
                   component_mapping="Unverified", created=time.time(),
                   settings={key: request[key] for key in ["logic_mode", "attempts", "seed", "condition"]},
                   generation_settings={"lattice_steps":30,"geometry_steps":50,"evaluation_threshold":0.6,"max_evaluate_num":2,
                       "designer":os.environ.get("METASYMBO_DESIGNER", "gpt-4o-mini"),
                       "supervisor":os.environ.get("METASYMBO_SUPERVISOR", "gpt-4.1")} if request["kind"] == "generation" else None,
                   **provenance))
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(path)
        identifier = hashlib.sha256(f"workspace/{path.name}".encode()).hexdigest()[:24]
        status["candidates"].append(identifier)
        update("Candidate saved")
        return identifier

    try:
        update("Loading model environment", state="running")
        import torch
        from modules.geom_autoencoder import GeomVAE, EncoderwithPredictionHead
        from modules.submodules import LatticeNormalizer
        from utils.llm_utils import classify_nodes_with_geometry

        random.seed(request["seed"])
        np.random.seed(request["seed"])
        torch.manual_seed(request["seed"])
        torch.set_num_threads(2)
        device = os.environ.get("METASYMBO_DEVICE", "cpu")

        if request["kind"] == "prediction":
            source = read_lattice(request["source_path"])
            args = dict(max_node_num=100, latent_dim=128, edge_sample_threshold=0.5,
                        is_variational=True, is_disent_variational=True, is_condition=True,
                        condition_dim=12, disentangle_same_layer=True)
            generator = GeomVAE(LatticeNormalizer(), **args).to(device)
            generator.load_state_dict(torch.load(checkpoint / "best_ae_model.pt", map_location=device, weights_only=True))
            predictor = EncoderwithPredictionHead(GeomVAE(LatticeNormalizer(), **args), 128, 12).to(device)
            predictor.load_state_dict(torch.load(checkpoint / "best_predictor_model.pt", map_location=device, weights_only=True))
            predictor.eval()
            coords = torch.tensor(source["frac_coords"], dtype=torch.float32, device=device)
            edges = torch.tensor(source["edge_index"], dtype=torch.long, device=device)
            lengths = torch.tensor([source["lengths"]], dtype=torch.float32, device=device)
            angles = torch.tensor([source["angles"]], dtype=torch.float32, device=device)
            node_types = classify_nodes_with_geometry(coords, edges).argmax(dim=1).long() + 1
            batch = torch.zeros(len(coords), dtype=torch.long, device=device)
            counts = torch.tensor([len(coords)], dtype=torch.long, device=device)
            update("Predicting elastic properties")
            with torch.no_grad():
                normalized_lengths, normalized_angles = generator.normalizer(lengths, angles)
                prediction = predictor(node_types, coords, edges, batch, normalized_lengths, normalized_angles, counts, denormalize=True)
            identifier = save_candidate(coords, edges, lengths, angles, prediction, geometry_fingerprint=source["fingerprint"])
            update("Prediction complete", state="complete", selected_candidate=identifier)
        else:
            from datasets import LatticeModulus
            from model.modal_agents import MetamatGenAgents

            class WorkbenchAgents(MetamatGenAgents):
                # Capture the existing workflow at its public collaboration
                # boundaries; do not maintain a second optimizer implementation.
                def count_call(self, stage):
                    self.ui_api_calls = getattr(self, "ui_api_calls", 0) + 1
                    if self.ui_api_calls > 12:
                        raise RuntimeError("Provider call budget exhausted")
                    update(stage)

                def load_translator(self):
                    return bounded_responses(super().load_translator(), self.count_call, "Understanding request and creating scaffold")

                def load_supervisor(self):
                    client = super().load_supervisor()
                    self.Predictor.register_forward_pre_hook(lambda *_: update("Predicting properties"))
                    return bounded_responses(client, self.count_call, "Supervisor evaluation")

                def translate(self, *args, **kwargs):
                    result = super().translate(*args, **kwargs)
                    _, coords, edges, _, lengths, angles, _ = result
                    buffer = io.BytesIO()
                    np.savez(buffer, frac_coords=coords.cpu().numpy(), edge_index=edges.cpu().numpy(),
                             lengths=lengths.cpu().numpy(), angles=angles.cpu().numpy())
                    geometry = read_lattice(buffer.getvalue())
                    if geometry["nodes"] > 100 or not geometry["struts"] or geometry["zero_length_struts"]:
                        raise ValueError("Generated scaffold is outside predictor limits")
                    return result

                def collaborate_between_agents_12(self, *args, **kwargs):
                    self.ui_trials = getattr(self, "ui_trials", 0) + 1
                    if self.ui_trials > 8:
                        raise RuntimeError("Geometry retry budget exhausted")
                    update("Generating structure")
                    return super().collaborate_between_agents_12(*args, **kwargs)

                def collaborate_between_agents_13(self, *args, **kwargs):
                    update("Understanding request")
                    return super().collaborate_between_agents_13(*args, **kwargs)

                def collaborate_between_agents_23(self, *args, **kwargs):
                    update("Generating structure")
                    result = super().collaborate_between_agents_23(*args, **kwargs)
                    _, coords, edges, lengths, angles, score, predictions, _, improved_prompt, _ = result
                    identifier = save_candidate(coords, edges, lengths, angles, predictions,
                                                supervisor_score=float(score), suggested_prompt=improved_prompt)
                    if score > getattr(self, "ui_best_score", -float("inf")):
                        self.ui_best_score = score
                        self.ui_best_id = identifier
                    return result

            update("Loading reference dataset")
            dataset = LatticeModulus(os.environ["METASYMBO_DATASET"], file_name="data")
            agents = WorkbenchAgents(root=ROOT, ckpt_dir=checkpoint, device=device,
                                     api_key=os.environ["OPENAI_API_KEY"],
                                     designer_client=os.environ.get("METASYMBO_DESIGNER", "gpt-4o-mini"),
                                     supervisor_client=os.environ.get("METASYMBO_SUPERVISOR", "gpt-4.1"),
                                     evaluation_threshold=0.6, max_evaluate_num=2)
            condition = request.get("condition")
            if condition is not None:
                condition = torch.tensor([condition], dtype=torch.float32, device=device)
            agents.collaborative_end_to_end_generation(
                dataset, request["prompt"], logic_mode=request["logic_mode"],
                max_collaboration_num=request["attempts"], num_steps_lattice=30,
                num_steps_geo=50, condition=condition, verbose=False, save_dir=None)
            reason = "Supervisor threshold reached" if agents.ui_best_score >= 0.6 else "Attempt limit reached"
            update(reason, state="complete", selected_candidate=agents.ui_best_id)
    except Exception as exc:
        # Avoid copying provider exceptions, request bodies, or credentials into
        # browser-readable status files. Show an actionable category instead.
        name = type(exc).__name__
        log_exception(LOGGER, status["id"], exc)
        if "Authentication" in name:
            message = "The design service is temporarily unavailable. Please try again later."
        elif "RateLimit" in name:
            message = "The design provider is busy or its allowance is used. Please try again later."
        elif "Connection" in name:
            message = "The design provider could not be reached. Please try again later."
        elif "Timeout" in name:
            message = "The design provider timed out. Please try again later."
        elif name in {"ModuleNotFoundError", "ImportError"}:
            message = "The model service is temporarily unavailable. Please try again later."
        elif "Supervisor" in status["stage"]:
            message = "The supervisor could not evaluate this candidate. Saved candidates are retained."
        elif "Predicting" in status["stage"]:
            message = "The predictor could not evaluate this geometry. Saved candidates are retained."
        elif name == "ValueError":
            message = "The generated geometry was invalid. Try a simpler design brief."
        else:
            message = "The model could not complete this run within the demo limits. Saved candidates are retained."
        update(message, state="failed", error_type=name)


if __name__ == "__main__":
    configure_logging()
    # Legacy model routines print provider output and prompts. Only our safe
    # structured logger reaches journald; its handler keeps the original stderr.
    with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        main(Path(sys.argv[1]).resolve())
