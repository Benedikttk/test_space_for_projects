"""Command-line interface for blackjack EV queries."""

from __future__ import annotations

from typing import List

import typer

from blackjack.ev import action_evs, best_action
from blackjack.hand import Hand
from blackjack.rules import RuleSet
from blackjack.shoe import Shoe

app = typer.Typer(help="Blackjack EV tools")


def _parse_cards(cards: str) -> List[str]:
    return [card.strip().upper() for card in cards.split(",") if card.strip()]


@app.command("recommend")
def recommend(
    player_cards: str = typer.Option(..., "--player", help="Comma-separated player cards, e.g. A,7"),
    dealer_upcard: str = typer.Option(..., "--dealer", help="Dealer upcard rank, e.g. 6"),
    decks: int = typer.Option(8, min=1, help="Number of decks in the shoe"),
) -> None:
    """Compute EV table and best action for one hand."""
    cards = _parse_cards(player_cards)
    if len(cards) < 2:
        raise typer.BadParameter("player must include at least two cards")

    hand = Hand(cards=cards)
    evs = action_evs(hand, dealer_upcard.upper(), Shoe(decks=decks), RuleSet())
    action, ev = best_action(evs)
    typer.echo(f"best_action={action} ev={ev:+.4f}")
    for name, value in sorted(evs.items(), key=lambda item: item[1], reverse=True):
        typer.echo(f"{name:>10}: {value:+.4f}")


if __name__ == "__main__":
    app()
