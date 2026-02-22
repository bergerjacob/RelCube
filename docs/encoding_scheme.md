# Rubik's Cube Piece Encoding Reference

## Slot Layout by Layer

### U Layer (8 pieces)
```
  ╔═══╦═══╦═══╗
  ║ 2 ║11 ║ 3 ║
  ║UBL║UB ║UBR║
  ╠═══╬═══╬═══╣
  ║10 ║   ║ 8 ║
  ║UL ║   ║UR ║
  ╠═══╬═══╬═══╣
  ║ 1 ║ 9 ║ 0 ║
  ║UFL║UF ║UFR║
  ╚═══╩═══╩═══╝
```

### E Layer (4 pieces - middle slice)
```
  ╔═══╦═══╦═══╗
  ║18 ║   ║19 ║
  ║BL ║   ║BR ║
  ╠═══╬═══╬═══╣
  ║   ║   ║   ║
  ║   ║   ║   ║
  ╠═══╬═══╬═══╣
  ║17 ║   ║16 ║
  ║FL ║   ║FR ║
  ╚═══╩═══╩═══╝
```

### D Layer (8 pieces)
```
  ╔═══╦═══╦═══╗
  ║ 6 ║15 ║ 7 ║
  ║DBL║DB ║DBR║
  ╠═══╬═══╬═══╣
  ║14 ║   ║12 ║
  ║DL ║   ║DR ║
  ╠═══╬═══╬═══╣
  ║ 5 ║13 ║ 4 ║
  ║DFL║DF ║DFR║
  ╚═══╩═══╩═══╝
```

## Full Slot Reference

| Slot | Piece | Layer |
|------|-------|-------|
| 0 | UFR | U |
| 1 | UFL | U |
| 2 | UBL | U |
| 3 | UBR | U |
| 4 | DFR | D |
| 5 | DFL | D |
| 6 | DBL | D |
| 7 | DBR | D |
| 8 | UR | U |
| 9 | UF | U |
| 10 | UL | U |
| 11 | UB | U |
| 12 | DR | D |
| 13 | DF | D |
| 14 | DL | D |
| 15 | DB | D |
| 16 | FR | E |
| 17 | FL | E |
| 18 | BL | E |
| 19 | BR | E |

## Orientation Encoding

### Corners (Slots 0-7, orientations 0-2)
Corners have 3 possible orientations:
- **Orientation 0**: U/D sticker on U/D face
- **Orientation 1**: U/D sticker on F/B face (clockwise twist)
- **Orientation 2**: U/D sticker on L/R face (counter-clockwise twist)

| Slot | Piece | Stickers (Primary, Face2, Face3) | Orient 0 | Orient 1 | Orient 2 |
|------|-------|----------------------------------|----------|----------|----------|
| 0 | UFR | (U, F, R) | UFR | FRU | RUF |
| 1 | UFL | (U, L, F) | UFL | LFU | FUL |
| 2 | UBL | (U, B, L) | UBL | BLU | LUB |
| 3 | UBR | (U, R, B) | UBR | RBU | BUR |
| 4 | DFR | (D, R, F) | DFR | RFD | FDR |
| 5 | DFL | (D, F, L) | DFL | FLD | LDF |
| 6 | DBL | (D, L, B) | DBL | LBD | BLD |
| 7 | DBR | (D, B, R) | DBR | BDR | RBD |

### Edges (Slots 8-19, orientations 0-1)
Edges have 2 possible orientations:
- **Orientation 0**: Primary sticker (U/D for U-layer edges, F/B for E-layer edges) on primary face
- **Orientation 1**: Primary sticker on secondary face (flip)

| Slot | Piece | Stickers (Primary, Face2) | Orient 0 | Orient 1 |
|------|-------|---------------------------|----------|----------|
| 8 | UR | (U, R) | UR | RU |
| 9 | UF | (U, F) | UF | FU |
| 10 | UL | (U, L) | UL | LU |
| 11 | UB | (U, B) | UB | BU |
| 12 | DR | (D, R) | DR | RD |
| 13 | DF | (D, F) | DF | FD |
| 14 | DL | (D, L) | DL | LD |
| 15 | DB | (D, B) | DB | BD |
| 16 | FR | (F, R) | FR | RF |
| 17 | FL | (F, L) | FL | LF |
| 18 | BL | (B, L) | BL | LB |
| 19 | BR | (B, R) | BR | RB |

## Encoding Format

The piece encoding produces three arrays of length 20:

1. **Slots**: Array of slot indices [0-19] indicating current piece positions
2. **Pieces**: Array of piece IDs [0-19] indicating which piece is at each slot
3. **Orientations**: Array of orientation values:
   - Corners (slots 0-7): values 0, 1, or 2
   - Edges (slots 8-19): values 0 or 1 |
