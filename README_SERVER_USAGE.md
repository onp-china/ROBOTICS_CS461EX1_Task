# Server Usage Guide (RTX 5090 + MuJoCo Baseline)

This guide is for our current remote server workflow:

- Host: `connect.westb.seetacloud.com`
- Port: `49753`
- User: `root`
- Project path: `/root/autodl-tmp/ROBOTICS_CS461EX1_Task`
- Branch: `mujoco`

---

## 1. Login from Mac

```bash
ssh -p 49753 root@connect.westb.seetacloud.com
```

After login, verify GPU:

```bash
nvidia-smi
```

---

## 2. Use tmux (must-use for long jobs)

Create a session:

```bash
tmux new -s mujoco
```

Detach but keep training running:

```bash
# press:
Ctrl+b, then d
```

Re-attach later:

```bash
tmux attach -t mujoco
```

List sessions:

```bash
tmux ls
```

Kill a session (careful: it stops jobs):

```bash
tmux kill-session -t mujoco
```

---

## 3. First-time environment setup (conda)

```bash
cd /root/autodl-tmp/ROBOTICS_CS461EX1_Task
conda create -n mujoco python=3.10 -y
conda activate mujoco
pip install -r requirements_mujoco.txt
```

If repo is missing:

```bash
cd /root/autodl-tmp
git clone https://github.com/onp-china/ROBOTICS_CS461EX1_Task.git
cd ROBOTICS_CS461EX1_Task
git checkout mujoco
```

---

## 4. Pull latest code

```bash
cd /root/autodl-tmp/ROBOTICS_CS461EX1_Task
git checkout mujoco
git pull origin mujoco
```

---

## 5. Run baseline experiment

```bash
cd /root/autodl-tmp/ROBOTICS_CS461EX1_Task
conda activate mujoco
python scripts/train.py --config configs/mujoco_push_a0_baseline.yaml
```

Other configs:

- `configs/mujoco_push_a1_stride1.yaml`
- `configs/mujoco_push_a2_seq12_stride1.yaml`
- `configs/mujoco_push_a3_seq12_stride1_attnres.yaml`

---

## 6. Disk usage tips (50GB data disk)

Check usage:

```bash
df -h
du -sh /root/autodl-tmp/ROBOTICS_CS461EX1_Task/results/* 2>/dev/null
```

Guideline:

- 50GB is enough for current MuJoCo D0 runs.
- Expand to 100GB if remaining space drops below 15GB.

---

## 7. About the “terminal control codes” paste warning

If your terminal shows:

`The text to be pasted contains terminal control codes...`

Use:

- **`Sanitize and paste`** (recommended)

Avoid:

- `Paste anyway` unless you fully trust the source and know why control sequences are needed.

Reason: copied text may contain hidden control characters that can execute unexpected commands.

---

## 8. Safe daily workflow

1. SSH login
2. `tmux attach -t mujoco` (or create one)
3. `conda activate mujoco`
4. `git pull origin mujoco`
5. Run training/evaluation
6. Detach `Ctrl+b d` before closing local terminal

---

## 9. Quick troubleshooting

Conda not found:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate mujoco
```

GPU not visible:

```bash
nvidia-smi
```

If this fails, restart instance or re-check selected CUDA image.

