#!/usr/bin/env python3
"""Apply risk agent equity calculation fix."""

def main():
    with open('src/agents/risk.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # Find and replace the equity calculation
    old_code = '''        # Perform quantitative risk calculations
        try:
            position_size = self._calculate_position_size(
                portfolio_state.get("equity", 0),
                market_data.price
            )'''

    new_code = '''        # Perform quantitative risk calculations
        try:
            # Calculate portfolio equity from cash and positions
            cash = portfolio_state.get("cash", 0.0)
            positions = portfolio_state.get("positions", {})
            positions_value = sum(
                pos.get("quantity", 0) * pos.get("current_price", 0)
                for pos in positions.values()
            )
            portfolio_equity = cash + positions_value

            position_size = self._calculate_position_size(
                portfolio_equity,
                market_data.price
            )'''

    if old_code in content:
        content = content.replace(old_code, new_code)

        with open('src/agents/risk.py', 'w', encoding='utf-8') as f:
            f.write(content)

        print("[OK] Applied risk agent equity calculation fix")
        print("   - Now calculates equity from cash + positions")
        print("   - Handles cash-only portfolios correctly")
    else:
        print("[ERROR] Could not find the code to replace")
        print("The file may have been modified already")

if __name__ == '__main__':
    main()
