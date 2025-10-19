"""
run_face.py
-----------
CLI mínima para testar a cadeia facial (com impressão agrupada).
"""

import argparse
from .face_sensor      import FaceSensor
from .face_encoder     import HeuristicFaceEncoder
from .face_recorder    import FaceRecorder
from .features         import PRINT_GROUPS, feature_names


def main():
    parser                    = argparse.ArgumentParser(description="Teste de captura/registro facial (MediaPipe).")
    parser.add_argument("--cam",           type=int,   default=0)
    parser.add_argument("--width",         type=int,   default=640)
    parser.add_argument("--height",        type=int,   default=480)
    parser.add_argument("--style",         type=str,   default="boxes_brows",
                        choices=["contours","oval","boxes","hud","boxes_brows"])
    parser.add_argument("--line-thick",    type=int,   default=1)
    parser.add_argument("--alpha",         type=float, default=0.90)
    parser.add_argument("--no-preview",    action="store_true")
    args                      = parser.parse_args()

    encoder                   = HeuristicFaceEncoder()
    recorder                  = FaceRecorder(out_dir="data/face")
    names                     = feature_names()

    # fmt: off
    def on_faceframe(ff):
        fe                    = encoder.encode(ff)
        recorder.write(fe)
        vals                  = {name: float(v) for name, v in zip(names, fe.features.tolist())}

        parts                 = [f"t={fe.t:.0f}"]
        for label, group, render in PRINT_GROUPS:
            if render is None:
                # imprime individualmente (ex.: mouth=0.12)
                for k in group:
                    parts.append(f"{label}={vals.get(k, 0.0):.2f}")
            else:
                parts.append(f"{label}={render(vals)}")

        print("  ".join(parts))
    # fmt: on

    sensor                    = FaceSensor(
        cam_index             = args.cam,
        width                 = args.width,
        height                = args.height,
        enable_preview        = (not args.no_preview),
        overlay_style         = args.style,
        line_thickness        = args.line_thick,
        overlay_alpha         = args.alpha,
    )
    sensor.on_frame(on_faceframe)

    try:
        sensor.start()
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
