# TXTure: ASCII ART with webcam

## Env Setup

```bash
uv venv
```

```bash
source .venv/bin/activate
```

```bash
uv pip install -e .
```

## Live ASCII Webcam

```bash
uv run txture-live
```

### Options

Show OpenCV preview window.

```bash
--preview
```

Enable hidden controller for live adjustments.

```bash
--control
```

Enable ANSI true-color ASCII output.

```bash
--color
```

NAME Choose charset (ascii_all, letters, digits, etc.).

```bash
--set
```

Target ASCII frame rate.

```bash
--fps N
```

## Exit

Ctrl + C

or focus on 'control' window and press 'ECS' (not focus on the terminal!)
