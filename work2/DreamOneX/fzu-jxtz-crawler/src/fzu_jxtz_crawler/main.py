"""
CLI entry point for the FZU announcement crawler.

This module provides a command-line interface with rich output.
"""
import argparse
import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .client import CrawlerClient
from .crawler import crawl_announcement_list, crawl_announcement_details
from .export import export_to_csv
from .models import AnnouncementEntry
from .url_utils import build_page_urls


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="FZU Academic Affairs Announcement Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch 500 announcements with full details, export to CSV (default)
  %(prog)s
  
  # Fetch only 10 announcements without details
  %(prog)s --count 10 --details 0
  
  # Fetch 20 announcements with details for first 5, custom CSV filename
  %(prog)s --count 20 --details 5 --export-csv my_data.csv
  
  # Fetch 50 announcements, no CSV export
  %(prog)s -n 50 --export-csv ""
  
  # Custom preview length
  %(prog)s --count 100 --preview-length 500
  
  # Download attachments (requires details)
  %(prog)s --count 10 --download-attachments ./downloads
        """
    )
    
    _ = parser.add_argument(
        "-c", "--concurrency",
        type=int,
        default=5,
        metavar="N",
        help="number of concurrent threads (default: 5)"
    )
    
    _ = parser.add_argument(
        "-n", "--count",
        type=int,
        default=500,
        metavar="N",
        help="number of announcements to fetch (default: 500)"
    )
    
    _ = parser.add_argument(
        "-d", "--details",
        type=int,
        default=-1,
        metavar="N",
        help="number of announcements to fetch full details for (default: -1 = all, 0 = none)"
    )
    
    _ = parser.add_argument(
        "-l", "--preview-length",
        type=int,
        default=200,
        metavar="N",
        help="maximum length of body preview in characters (default: 200)"
    )
    
    _ = parser.add_argument(
        "-a", "--show-attachments",
        action="store_true",
        help="show attachment information (default: only when fetching details)"
    )
    
    _ = parser.add_argument(
        "--download-attachments",
        type=str,
        metavar="DIR",
        help="download attachments to specified directory"
    )
    
    _ = parser.add_argument(
        "--export-csv",
        type=str,
        default="announcements.csv",
        metavar="FILE",
        help="export announcements to CSV file (default: announcements.csv)"
    )
    
    _ = parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="quiet mode, minimal output"
    )
    
    _ = parser.add_argument(
        "--no-table",
        action="store_true",
        help="skip displaying the announcement table"
    )
    
    return parser.parse_args()


def fetch_announcements(
    client: CrawlerClient,
    count: int,
    console: Console,
    quiet: bool = False,
    max_workers: int = 5
) -> list[AnnouncementEntry]:
    """
    Fetch the specified number of announcements.
    
    Args:
        client: CrawlerClient instance
        count: Number of announcements to fetch
        console: Rich console for output
        quiet: Whether to suppress progress output
        max_workers: Number of threads to use
        
    Returns:
        List of AnnouncementEntry objects
    """
    # Get total pages
    total_pages = client.get_total_pages()
    max_page = total_pages - 1
    
    if not quiet:
        console.print(f"[green]✓[/green] Found [bold]{total_pages}[/bold] total pages")
    
    # Fetch pages until we have enough announcements
    all_entries: list[AnnouncementEntry] = []
    current_page = max_page
    
    def on_page_fetched(_idx: int, _total: int, _entry_count: int) -> None:
        if not quiet:
            # Note: This might be called from multiple threads, but rich.print is thread-safe enough usually.
            # However, simpler to just show overall progress bar updates or rely on the main progress bar.
            # Printing lines might get interleaved. 
            pass
            # console.print(f"[green]✓[/green] Page {max_page - current_page + idx}/{total}: Found {entry_count} announcements")
    
    while len(all_entries) < count and current_page >= 0:
        # Calculate how many pages we might need
        # Assume ~10 announcements per page
        estimated_pages_needed = max(1, (count - len(all_entries) + 9) // 10)
        end_page = max(0, current_page - estimated_pages_needed + 1)
        
        urls = build_page_urls(max_page, start_page=current_page, end_page=end_page)
        
        if not quiet:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                # We fetch a batch of pages. 
                task = progress.add_task(f"[cyan]Fetching pages {end_page}-{current_page} with {max_workers} threads...", total=None)
                page_entries = crawl_announcement_list(client, urls, on_page_fetched, max_workers=max_workers)
                all_entries.extend(page_entries)
                progress.update(task, completed=True)
        else:
            page_entries = crawl_announcement_list(client, urls, max_workers=max_workers)
            all_entries.extend(page_entries)
        
        current_page = end_page - 1
    
    # Trim to exact count
    return all_entries[:count]


def display_announcement_table(
    entries: list[AnnouncementEntry],
    console: Console,
    max_display: int = 10
) -> None:
    """
    Display announcements in a table format.
    
    Args:
        entries: List of announcements to display
        console: Rich console for output
        max_display: Maximum number of entries to display
    """
    table = Table(title=f"Announcements Overview (showing {min(len(entries), max_display)} of {len(entries)})", border_style="cyan")
    table.add_column("#", style="cyan", width=4)
    table.add_column("Title", style="white")
    table.add_column("Issuer", style="yellow", width=20)
    table.add_column("Date", style="green", width=12)
    
    for i, entry in enumerate(entries[:max_display], 1):
        table.add_row(
            str(i),
            entry.title,
            entry.issuer,
            entry.pub_date.strftime('%Y-%m-%d')
        )
    
    console.print(table)
    console.print()


def display_announcement_details(
    entry: AnnouncementEntry,
    index: int,
    console: Console,
    preview_length: int = 200,
    show_attachments: bool = True
) -> None:
    """
    Display detailed information for a single announcement.
    
    Args:
        entry: Announcement entry to display
        index: Index number for display
        console: Rich console for output
        preview_length: Maximum length of body preview
        show_attachments: Whether to show attachment information
    """
    console.print(f"[bold cyan]━━━ Announcement {index} ━━━[/bold cyan]")
    console.print(f"[bold]Title:[/bold] {entry.title}")
    console.print(f"[bold]Issuer:[/bold] {entry.issuer}")
    console.print(f"[bold]Date:[/bold] {entry.pub_date.strftime('%Y-%m-%d')}")
    console.print(f"[bold]URL:[/bold] [link]{entry.url}[/link]")
    
    # Display body preview
    if entry.body:
        body_preview = entry.body[:preview_length].strip()
        if len(entry.body) > preview_length:
            body_preview += "..."
        console.print(f"\n[bold]Body:[/bold]\n{body_preview}")
    
    # Display attachments
    if show_attachments:
        if entry.attachments:
            console.print(f"\n[bold]Attachments ({len(entry.attachments)}):[/bold]")
            for att in entry.attachments:
                console.print(f"  [cyan]•[/cyan] {att.name} [dim](downloads: {att.download_times})[/dim]")
                console.print(f"    [dim]{att.url}[/dim]")
        else:
            console.print("\n[bold]Attachments:[/bold] [dim]None[/dim]")
    
    console.print()


def download_attachments(
    _entries: list[AnnouncementEntry],
    download_dir: str,
    console: Console,
    quiet: bool = False
) -> None:
    """
    Download attachments from announcements.
    
    Args:
        _entries: List of announcements with attachments (unused, for future implementation)
        download_dir: Directory to save attachments
        console: Rich console for output
        quiet: Whether to suppress output
    """
    # TODO: Implement attachment downloading
    # This would require adding download functionality to the client
    if not quiet:
        console.print(f"[yellow]⚠[/yellow] Attachment downloading not yet implemented")
        console.print(f"[dim]Would download to: {download_dir}[/dim]")
        console.print()


def main(args: argparse.Namespace | None = None) -> int:
    """
    Main CLI function.
    
    Args:
        args: Parsed arguments (if None, will parse from sys.argv)
        
    Returns:
        Exit code (0 for success, 1 for error)
    """
    if args is None:
        args = parse_args()
    
    console = Console()
    err_console = Console(stderr=True)
    
    # Validate arguments
    if args.count <= 0:  # pyright: ignore[reportAny]
        err_console.print("[red]Error:[/red] --count must be positive")
        return 1
    
    if args.preview_length < 0:  # pyright: ignore[reportAny]
        err_console.print("[red]Error:[/red] --preview-length must be non-negative")
        return 1

    if args.concurrency < 1:  # pyright: ignore[reportAny]
         err_console.print("[red]Error:[/red] --concurrency must be at least 1")
         return 1
    
    # Calculate how many details to fetch
    details_count = args.details  # pyright: ignore[reportAny]
    if details_count == -1:
        details_count = args.count  # pyright: ignore[reportAny]
    elif details_count > args.count:  # pyright: ignore[reportAny]
        details_count = args.count  # pyright: ignore[reportAny]
    
    # Print header
    if not args.quiet:  # pyright: ignore[reportAny]
        console.print(Panel.fit(
            f"[bold cyan]FZU Academic Affairs Announcement Crawler[/bold cyan]\n[dim]Concurrency: {args.concurrency} threads[/dim]",  # pyright: ignore[reportAny]
            border_style="cyan"
        ))
        console.print()
    
    try:
        with CrawlerClient() as client:
            # Step 1: Fetch announcements
            if not args.quiet:  # pyright: ignore[reportAny]
                console.print(f"[cyan]→[/cyan] Fetching [bold]{args.count}[/bold] announcements...")  # pyright: ignore[reportAny]
                console.print()
            
            all_entries = fetch_announcements(client, args.count, console, args.quiet, max_workers=args.concurrency)  # pyright: ignore[reportAny]
            
            if not args.quiet:  # pyright: ignore[reportAny]
                console.print()
                console.print(f"[green]✓[/green] Fetched [bold]{len(all_entries)}[/bold] announcements")
                console.print()
            
            # Step 2: Display table
            if not args.no_table and not args.quiet and all_entries:  # pyright: ignore[reportAny]
                display_announcement_table(all_entries, console)
            
            # Step 3: Fetch and display details
            if details_count > 0:
                entries_to_detail = all_entries[:details_count]
                
                if not args.quiet:  # pyright: ignore[reportAny]
                    console.print(Panel.fit(
                        f"[bold cyan]Fetching Details for {len(entries_to_detail)} Announcements[/bold cyan]",
                        border_style="cyan"
                    ))
                    console.print()
                
                # Fetch all details in batch with concurrency
                if not args.quiet:  # pyright: ignore[reportAny]
                    with Progress(
                        SpinnerColumn(),
                        TextColumn("[progress.description]{task.description}"),
                        console=console
                    ) as progress:
                        task = progress.add_task(f"[cyan]Fetching details... ({args.concurrency} threads)", total=None)  # pyright: ignore[reportAny]
                        crawl_announcement_details(client, entries_to_detail, max_workers=args.concurrency)  # pyright: ignore[reportAny]
                        progress.update(task, completed=True)
                else:
                    crawl_announcement_details(client, entries_to_detail, max_workers=args.concurrency)  # pyright: ignore[reportAny]
                
                # Display loop (fetching already done above)
                for i, entry in enumerate(entries_to_detail, 1):
                    if not args.quiet:  # pyright: ignore[reportAny]
                        display_announcement_details(
                            entry,
                            i,
                            console,
                            preview_length=args.preview_length,  # pyright: ignore[reportAny]
                            show_attachments=True
                        )
            
            # Step 4: Show attachments for entries without details (if requested)
            elif args.show_attachments and not args.quiet:  # pyright: ignore[reportAny]
                console.print(Panel.fit(
                    "[bold cyan]Attachment Information[/bold cyan]",
                    border_style="cyan"
                ))
                console.print()
                console.print("[yellow]Note:[/yellow] Use --details to fetch attachment information")
                console.print()
            
            # Step 5: Download attachments (if requested)
            if args.download_attachments:  # pyright: ignore[reportAny]
                entries_with_details = all_entries[:details_count] if details_count > 0 else []
                if entries_with_details:
                    download_attachments(entries_with_details, args.download_attachments, console, args.quiet)  # pyright: ignore[reportAny]
                elif not args.quiet:  # pyright: ignore[reportAny]
                    console.print("[yellow]⚠[/yellow] No details fetched, cannot download attachments")
                    console.print()
            
            # Step 6: Export to CSV (if requested)
            if args.export_csv:  # pyright: ignore[reportAny]
                if not args.quiet:  # pyright: ignore[reportAny]
                    console.print(f"[cyan]→[/cyan] Exporting to CSV: [bold]{args.export_csv}[/bold]...")  # pyright: ignore[reportAny]
                
                export_to_csv(all_entries, args.export_csv, clean_text_fields=True)  # pyright: ignore[reportAny]
                
                if not args.quiet:  # pyright: ignore[reportAny]
                    console.print(f"[green]✓[/green] Exported [bold]{len(all_entries)}[/bold] announcements to CSV")
                    console.print()
            
            # Success message
            if not args.quiet:  # pyright: ignore[reportAny]
                console.print(Panel.fit(
                    "[bold green]✓ Completed Successfully![/bold green]",
                    border_style="green"
                ))
            
            return 0
            
    except KeyboardInterrupt:
        if not args.quiet:  # pyright: ignore[reportAny]
            console.print("\n[yellow]⚠[/yellow] Interrupted by user")
        return 130
    except Exception as e:
        err_console.print(f"[red]✗ Error:[/red] {e}")
        if not args.quiet:  # pyright: ignore[reportAny]
            import traceback
            err_console.print("\n[dim]" + traceback.format_exc() + "[/dim]")
        return 1


if __name__ == "__main__":
    sys.exit(main())
