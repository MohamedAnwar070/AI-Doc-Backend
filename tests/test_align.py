def _align_spatially(items):
    if not items:
        return []
    
    # Sort by top Y coordinate
    items.sort(key=lambda x: x[1][1] if x[1] else 0)
    
    lines = []
    current_line = []
    current_y_bottom = -1

    for text, bbox in items:
        if not bbox:
            current_line.append((text, [0,0,0,0]))
            continue
            
        y0 = bbox[1]
        y1 = bbox[3]
        
        # If this item overlaps vertically with the current line, add it
        if not current_line:
            current_line.append((text, bbox))
            current_y_bottom = y1
        else:
            # Overlap condition: y0 is less than the current line's average or max bottom
            # Actually, a simple threshold: if y0 is less than current_y_bottom, they are on the same line
            if y0 < current_y_bottom:
                current_line.append((text, bbox))
                current_y_bottom = max(current_y_bottom, y1)
            else:
                # New line
                lines.append(current_line)
                current_line = [(text, bbox)]
                current_y_bottom = y1
                
    if current_line:
        lines.append(current_line)
        
    aligned_text_lines = []
    for line in lines:
        # Sort items in the line by X coordinate
        line.sort(key=lambda x: x[1][0] if x[1] else 0)
        # Join them with a delimiter (e.g., ' \t ')
        line_text = " \t ".join(item[0] for item in line)
        aligned_text_lines.append(line_text)
        
    return aligned_text_lines

items = [
    ("Welcome To DigiGov", [0.1, 0.1, 0.3, 0.15]),
    ("Tamil Nadu", [0.5, 0.1, 0.7, 0.12]),
    ("Registration Department", [0.5, 0.13, 0.7, 0.15]),
    ("User Id", [0.1, 0.2, 0.2, 0.25]),
    ("591546817444", [0.3, 0.2, 0.5, 0.25]),
    ("Password", [0.1, 0.3, 0.2, 0.35]),
    ("xxxxxx", [0.3, 0.3, 0.5, 0.35]),
]

print("\n".join(_align_spatially(items)))
