# GGSort

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
./bin/ggsort <LOCATION> [IMAGE_DATA_DIR]
```
**Example:**
```bash
./bin/ggsort Waynes
```


### Navigation
- **SPACE** or **RIGHT ARROW**: Next image
- **LEFT ARROW**: Previous image
- **F**: Fast forward 10 images
- **B**: Fast backward 10 images
- **0–9** then **Shift+J**: Jump to a specific frame number (vim-style; ESC clears the buffer)
- **ESC** or **Q**: Exit viewer (ESC inside a mode cancels that mode first)

### Detection selection
- **Hover mouse**: Select the detection under the cursor (highlighted in white)
- **TAB**: When the mouse is inside multiple overlapping rectangles, cycle through them to pick the active one

### Detection editing
- **BACKSPACE** or **X**: Toggle selected detection as deleted
- **Z**: Mark **all** detections in the current image as deleted
- **Shift+Z**: Mark all as deleted **and** advance to the next image
- **H**: Toggle the "hard" flag on the selected detection
- **N**: Insert a new detection (see *Insert mode* below)
- **C**: Relocate the selected detection (see *Relocate mode* below)
- **D** (or click on a corner knob): Drag a corner of the selected detection to resize. Press **D** again or click to finish, or **ESC** to cancel and revert.

### Category assignment
Single keys assign a category to the **selected** detection; the Shift variant assigns to **every** detection in the current image.

| Key | Shift variant | Category |
|-----|---------------|----------|
| `g` | `G` | Gang-gang |
| `e` | —   | Person (mnemonic: p**e**rson) |
| `p` | `P` | Possum |
| `o` | `O` | Other |
| `v` | —   | Shortcut: set **all** detections to Other |

### Autodelete templates
- **Shift+K**: Save the selected detection as an autodelete template — future detections matching this template's location & size are auto-deleted on image load.
- **Shift+R**: Clear all saved autodelete templates.

### Insert mode (press **N**)

A two-step interactive flow for adding a new detection.

1. Press **N**. A high-contrast crosshair (inverted-pixel stripes flanked by black borders) extends from the mouse cursor rightward and downward, letting you align the next corner with image content. The top-right of the window shows `Insert: top-left`.
2. Move the cursor to the desired **top-left** corner, then press **SPACE** or **click** to lock it. A yellow dot marks the locked corner and the crosshair disappears.
3. Move the cursor to the desired **bottom-right** corner. A live preview rectangle (white, black-edged) tracks between the locked corner and the cursor. The top-right shows `Insert: bottom-right`.
4. Press **SPACE** or **click** again to commit. The new detection is created with the majority category of the other (non-deleted) detections in the image, or *Other* if there are none.

Press **ESC** at any point to cancel — nothing is added. Pressing **N** a second time toggles the mode off.

### Relocate mode (press **C**)

Same flow as Insert mode, but reused to move an existing detection.

1. Hover over a detection so it's selected, then press **C**. The selected detection is drawn black to show it's the one being moved, and the same crosshair appears.
2. Move the cursor to the new top-left, press **SPACE** or **click**.
3. Move to the new bottom-right, press **SPACE** or **click**.

Press **ESC** at any point to cancel — the detection stays at its original location. Pressing **C** a second time also cancels.

## Detection Categories

The system supports five categories with color-coded bounding boxes:

| Category | Color | Description |
|----------|-------|-------------|
| Gang-gang | Red | Gang-gang Cockatoos |
| Person | Blue | Human detections |
| Vehicle | Green | Vehicles |
| Possum | Green | Possums |
| Other | Turquoise | Other wildlife |

Deleted detections are shown in grey.
