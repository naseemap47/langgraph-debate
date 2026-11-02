import argparse
import asyncio
import sys
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from debate.utils.cdp_check import check_cdp_status
from debate.drivers.browser import BrowserCDPDriver
from debate.graph.graph import build_debate_graph
from debate.graph.state import DebateState
from debate.utils.exporter import export_debate_to_markdown

console = Console()

def run_check_cdp(args):
    status = check_cdp_status(args.endpoint)
    table = Table(title="Chrome DevTools Protocol (CDP) Status")
    table.add_column("Property", style="bold")
    table.add_column("Value")

    table.add_row("Connected", "[green]Yes[/green]" if status["connected"] else "[red]No[/red]")
    table.add_row("Endpoint", status["endpoint"])
    table.add_row("Open Tabs", str(status.get("raw_tab_count", 0)))

    for model, open_state in status.get("tabs", {}).items():
        val = "[green]Open[/green]" if open_state else "[yellow]Not detected[/yellow]"
        table.add_row(f"Tab: {model.capitalize()}", val)

    console.print(table)
    if not status["connected"]:
        console.print("\n[yellow]Tip:[/yellow] Run [bold]./scripts/run_chrome.sh[/bold] to launch Chrome with remote debugging.")

def run_debate_cli(args):
    models = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    judge = args.judge.strip().lower()

    console.print(Panel(f"[bold cyan]Starting Debate[/bold cyan]\nTopic: {args.topic}\nModels: {', '.join(models)}\nJudge: {judge}"))

    drivers = {m: BrowserCDPDriver(m, cdp_url=args.endpoint) for m in models}
    graph = build_debate_graph(drivers)

    initial_state: DebateState = {
        "topic": args.topic,
        "active_models": models,
        "judge_model": judge,
        "current_stage": "init",
        "stage_1_answers": {},
        "stage_2_critiques": {},
        "stage_3_synthesis": None,
        "status_events": [],
        "errors": {},
    }

    async def execute():
        with console.status("[bold green]Running debate through LangGraph..."):
            return await graph.ainvoke(initial_state)

    result = asyncio.run(execute())

    if result.get("stage_3_synthesis"):
        console.print(Panel(Markdown(result["stage_3_synthesis"]), title=f"[bold green]Final Synthesis (Judge: {result.get('judge_model')})[/bold green]"))

    if args.export:
        filepath = export_debate_to_markdown(result)
        console.print(f"[bold green]Saved debate transcript to:[/bold green] {filepath}")

def main():
    parser = argparse.ArgumentParser(description="LLM Debate Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_p = subparsers.add_parser("check-cdp", help="Check Chrome CDP connection and tabs")
    check_p.add_argument("--endpoint", default="http://127.0.0.1:9222", help="CDP endpoint")
    check_p.set_defaults(func=run_check_cdp)

    run_p = subparsers.add_parser("run", help="Run a multi-model debate")
    run_p.add_argument("topic", help="Topic to debate")
    run_p.add_argument("--models", default="claude,chatgpt,gemini,deepseek", help="Comma-separated models")
    run_p.add_argument("--judge", default="claude", help="Model to synthesize final consensus")
    run_p.add_argument("--endpoint", default="http://127.0.0.1:9222", help="CDP endpoint")
    run_p.add_argument("--export", action="store_true", help="Export report to markdown")
    run_p.set_defaults(func=run_debate_cli)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
