"""Per-video pose quality for the VAL split (Probe B): mean RTMPose hand confidence, fraction of
frames with a missing hand (mean conf < 0.3), from the raw keypoint pkls. CPU only."""
import json, pickle as pkl, sys
import numpy as np
ds_pose = {"ph": "/home/haipd/SLR/third_party/SEDS/PHOENIX-2014-T/RTM_Keypoints",
           "csl": "/home/haipd/SLR/third_party/SEDS/CSL/RTM_Keypoints",
           "h2s": "/home/haipd/SLR/third_party/SEDS/How2Sign/RTMpose/Pose_all_24rates"}
ds, split = sys.argv[1], sys.argv[2]
val = pkl.load(open(f"{split}/data/test.pkl", "rb"))
rows = []
for k, v in val.items():
    for e in (v if isinstance(v, list) else [v]):
        name = e.get("new_video_name", e["video_name"])
        d = pkl.load(open(f"{ds_pose[ds]}/{name}.pkl", "rb"))
        c = np.asarray(d["keypoints"])[:, :, 2]
        lh, rh = c[:, 91:112].mean(1), c[:, 112:133].mean(1)
        rows.append({"key": k, "video": name, "frames": int(c.shape[0]), "hand_conf": float(np.mean((lh + rh) / 2)),
                     "right_conf": float(rh.mean()), "left_conf": float(lh.mean()),
                     "missing_right_frac": float((rh < 0.3).mean()), "missing_left_frac": float((lh < 0.3).mean())})
json.dump(rows, open(f"{split}/pose_quality.json", "w"))
hc = np.array([r["hand_conf"] for r in rows]); print(ds, len(rows), "hand_conf quartiles", np.percentile(hc, [5, 25, 50, 75, 95]).round(3))
