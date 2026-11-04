"""Checks the GPU machine before serve.py: imports, checkpoint, backbone, prior. Prints "All checks passed"."""

import json
import socket
import sys

import paths

failed = False


def ok(msg: str) -> None:
    print(f"  ok    {msg}")


def bad(msg: str) -> None:
    global failed
    failed = True
    print(f"  FAIL  {msg}")


def main() -> None:
    try:
        import gr00t.policy.server_client  # noqa: F401
        import robot_ood.servo.session  # noqa: F401
        ok(f"imports ({sys.executable})")
    except Exception as error:
        bad(f"imports: {error}  (activate external/Isaac-GR00T/.venv)")

    config = paths.CHECKPOINT / "config.json"
    if not config.exists():
        bad(f"checkpoint: no {config}  (set VRH31_CKPT to the checkpoint directory)")
    else:
        ok(f"checkpoint {paths.CHECKPOINT}")

    meta = paths.PRIOR / "servo_prior.json"
    if not meta.exists():
        bad(f"prior: no {meta}  (run hf sync {paths.BUCKET}/vrh31_waist/prior {paths.PRIOR})")
    else:
        info = json.loads(meta.read_text())
        if info["checkpoint"] == paths.checkpoint_id(paths.CHECKPOINT):
            ok(f"prior {paths.PRIOR}: {info['n_bank']} starts, tau {info['tau']:.2f}")
        else:
            bad(f"prior was built on another checkpoint  (run hf sync {paths.BUCKET}/vrh31_waist/prior {paths.PRIOR})")

    with socket.socket() as s:
        if s.connect_ex(("127.0.0.1", paths.PORT)) == 0:
            bad(f"port {paths.PORT} is already in use (another serve.py?)")
        else:
            ok(f"port {paths.PORT} free")

    print("\nSome checks FAILED" if failed else "\nAll checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
