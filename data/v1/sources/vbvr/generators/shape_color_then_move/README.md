# O-11: Shape Color Then Move Data Generator

Generates synthetic visual reasoning tasks demonstrating sequential transformations: first color change, then position movement. The task presents a two-row analogy (A→B→C :: D→?→?) where shapes undergo the same two-step transformation pattern.

Each sample pairs a **task** (first frame + prompt describing what needs to happen) with its **ground truth solution** (final frame showing the result + video demonstrating how to achieve it). This structure enables both model evaluation and training.

---

## 📌 Basic Information

| Property | Value |
|----------|-------|
| **Task ID** | O-11 |
| **Task** | Shape Color Then Move |
| **Category** | Abstraction |
| **Resolution** | 1024×1024 px |
| **FPS** | 16 fps |
| **Duration** | ~3.75 seconds |
| **Output** | PNG images + MP4 video |

---

## 🚀 Usage

### Installation

```bash
# Clone the repository
git clone https://github.com/VBVR-DataFactory/O-11_shape_color_then_move_data-generator.git
cd O-11_shape_color_then_move_data-generator

# Install dependencies
pip install -r requirements.txt
```

### Generate Data

```bash
# Generate 100 samples
python examples/generate.py --num-samples 100

# Generate with specific seed
python examples/generate.py --num-samples 100 --seed 42

# Generate without videos
python examples/generate.py --num-samples 100 --no-videos

# Custom output directory
python examples/generate.py --num-samples 100 --output data/my_output
```

### Command-Line Options

| Argument | Type | Description | Default |
|----------|------|-------------|---------|
| `--num-samples` | int | Number of samples to generate | 100 |
| `--seed` | int | Random seed for reproducibility | Random |
| `--output` | str | Output directory | data |
| `--no-videos` | flag | Skip video generation | False |

---

## 📖 Task Example

### Prompt

```
The scene shows a sequential analogy A→B→C :: D→?→? with three shapes per row. On the top row, a color_238 minus becomes color_089, then moves down. On the bottom row, apply the same two-step transformation to the star: first change color from color_238 to color_089, then move down. Show both transformations sequentially in the video.
```
### Visual

<table>
<tr>
  <td align="center"><img src="samples/O-11_first_0.png" width="250"/></td>
  <td align="center"><img src="samples/O-11_video_0.gif" width="320"/></td>
  <td align="center"><img src="samples/O-11_final_0.png" width="250"/></td>
</tr>
<tr>
  <td align="center"><b>Initial Frame</b><br/>Two-row analogy with shapes</td>
  <td align="center"><b>Animation</b><br/>Color changes then position moves</td>
  <td align="center"><b>Final Frame</b><br/>Completed transformation sequence</td>
</tr>
</table>

---

## 📖 Task Description

### Objective

Complete a visual analogy by applying a two-step sequential transformation: first changing color, then changing position. The bottom row shape must follow the same transformation pattern demonstrated in the top row.

### Task Setup

- **Two-Row Analogy**: Top row shows example transformation (A→B→C), bottom row requires completion (D→?→?)
- **Sequential Transformations**: Two distinct steps applied in order (color first, then movement)
- **Color Change**: Shape changes from one color to another (e.g., red to blue) - this generator modifies the **color** property first, then the **position** parameter, demonstrating sequential modifications of both color and spatial properties
- **Position Movement**: Shape moves vertically (up or down) after color change
- **Shape Variation**: Different shapes in top and bottom rows (e.g., circle vs square)

### Key Features

- **Sequential reasoning**: Tests ability to recognize and apply multi-step transformations in correct order
- **Transformation decomposition**: Separates color and position changes into distinct steps
- **Pattern generalization**: Applies learned pattern from one shape to a different shape
- **Temporal ordering**: Color change occurs before position movement
- **Visual analogy**: A→B→C :: D→?→? structure tests analogical reasoning
- **Cross-attribute transformation**: Combines changes in both appearance (color) and spatial properties (position)
- **Large-scale diversity**: 20 distinct shapes, 400 colors, and 11 movement directions for extensive variety (6.6B+ unique combinations)

---

## 📦 Data Format

```
data/questions/shape_color_then_move_task/shape_color_then_move_00000000/
├── first_frame.png      # Initial state (two-row analogy setup)
├── final_frame.png      # Final state (completed transformation)
├── prompt.txt           # Task instructions
├── ground_truth.mp4     # Solution video (16 fps)
└── question_metadata.json # Task metadata
```


**File specifications**: Images are 1024×1024 PNG. Videos are MP4 at 16 fps, approximately 3.75 seconds long.

---

## 🏷️ Tags

`sequential-transformation` `visual-analogy` `color-change` `position-movement` `multi-step-reasoning` `pattern-recognition` `analogical-reasoning`

---
