#!/usr/bin/env python3
"""Apply portfolio agent fix."""

def main():
    with open('src/communication/orchestrator.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the line with "return {"final_decision": final_decision}"
    new_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is the portfolio management return statement
        if 'return {"final_decision": final_decision}' in line:
            # Check if we're in portfolio_management method (look back for method definition)
            in_portfolio_method = False
            for j in range(max(0, i-20), i):
                if 'async def run_portfolio_management' in lines[j]:
                    in_portfolio_method = True
                    break

            if in_portfolio_method:
                # Add the fix
                indent = ' ' * 8  # 8 spaces for method body
                new_lines.append(f'{indent}\n')
                new_lines.append(f'{indent}# Add portfolio decision to decisions dict so it appears in API response\n')
                new_lines.append(f'{indent}decisions["portfolio"] = final_decision\n')
                new_lines.append(f'{indent}\n')
                new_lines.append(f'{indent}return {{\n')
                new_lines.append(f'{indent}    "decisions": decisions,\n')
                new_lines.append(f'{indent}    "final_decision": final_decision\n')
                new_lines.append(f'{indent}}}\n')
                i += 1  # Skip the original return line
                continue

        new_lines.append(line)
        i += 1

    # Write back
    with open('src/communication/orchestrator.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("[OK] Applied portfolio agent fix")
    print("   - Added portfolio decision to decisions dict")
    print("   - Returns both decisions and final_decision")

if __name__ == '__main__':
    main()
