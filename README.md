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

## Interactive Controls

### Navigation
- **SPACE** or **RIGHT ARROW**: Next image
- **LEFT ARROW**: Previous image
- **ESC** or **Q**: Exit viewer

### Detection Management
- **Hover mouse**: Select detection (highlighted in white)
- **BACKSPACE** or **X**: Toggle selected detection as deleted
- **P**: Mark selected detection as Possum
- **O**: Mark selected detection as Other
- **G**: Mark selected detection as Gang-gang
- **C**: Relocate selected detection

When 'C' is pressed to relocate a highlighted detection, the detection will become dark grey. In this mode, move the mouse cursor to the top-left corner of the new detection rectangle and click the left button. A yellow marker will appear to indicate the selected location. Next, move the mouse to the bottom right corner of the new rectangle and click the left button a second time to finish the relocation. Press 'C' any time in this mode to cancel the relocation operation.

## Detection Categories

The system supports five categories with color-coded bounding boxes:

| Category | Color | Description |
|----------|-------|-------------|
| Gang-gang | Red | Gang-gang Cockatoos |
| Person | Blue | Human detections |
| Vehicle | Green | Vehicles |
| Possum | Yellow | Possums |
| Other | Turquoise | Other wildlife |

Deleted detections are shown in grey.
