"""
Main entry point for Brent Oil Change Point Detection CLI.
"""

import argparse
import sys
from pathlib import Path
import logging
import json
from typing import Optional

from config import AppConfig
from data_loader import DataLoader
from change_point import ChangePointDetector, ChangePointMethod
from risk_metrics import RiskAnalyzer
from visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Brent Oil Change Point Detection and Risk Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --data data/brent.csv --detect
  python main.py --data data/brent.csv --analyze-risk
  python main.py --data data/brent.csv --dashboard
  python main.py --data data/brent.csv --all --output results/
        """
    )

    parser.add_argument(
        '--data',
        type=str,
        default='data/brent.csv',
        help='Path to Brent oil price data CSV'
    )

    parser.add_argument(
        '--detect',
        action='store_true',
        help='Detect change points in the data'
    )

    parser.add_argument(
        '--analyze-risk',
        action='store_true',
        help='Calculate risk metrics'
    )

    parser.add_argument(
        '--dashboard',
        action='store_true',
        help='Generate visualization dashboard'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analyses'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory for results'
    )

    parser.add_argument(
        '--method',
        type=str,
        choices=['pelt', 'binary_seg', 'window', 'dynp'],
        default='pelt',
        help='Change point detection method'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def setup_output_directory(output_dir: Path) -> Path:
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def save_results(data: dict, output_path: Path, filename: str):
    """Save results to JSON file."""
    filepath = output_path / filename

    # Convert non-serializable objects
    def json_serializer(obj):
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(filepath, 'w') as f:
        json.dump(data, f, default=json_serializer, indent=2)

    logger.info(f"Saved results to {filepath}")


def main():
    """Main execution function."""
    args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = AppConfig()

    # Override data path if provided
    if args.data:
        config.data.data_path = Path(args.data)

    logger.info(f"Starting Brent Oil Analysis")
    logger.info(f"Configuration: {config}")

    try:
        # Load data
        loader = DataLoader(config.data)
        data = loader.load_data()

        logger.info(f"Loaded {len(data)} rows of data")
        logger.info(f"Date range: {data.index[0]} to {data.index[-1]}")

        # Create output directory
        output_dir = setup_output_directory(Path(args.output))

        # Save data summary
        stats = loader.get_summary_statistics()
        save_results(stats, output_dir, 'data_summary.json')

        # Determine what to run
        run_all = args.all or not (args.detect or args.analyze_risk or args.dashboard)

        # Initialize visualizer
        visualizer = Visualizer()

        # Change point detection
        change_points = []
        if run_all or args.detect:
            logger.info("Running change point detection...")

            # Map method string to enum
            method_map = {
                'pelt': ChangePointMethod.PELT,
                'binary_seg': ChangePointMethod.BINARY_SEG,
                'window': ChangePointMethod.WINDOW_SLIDING,
                'dynp': ChangePointMethod.DYNAMIC_PROGRAMMING
            }
            method = method_map.get(args.method, ChangePointMethod.PELT)

            detector = ChangePointDetector(config.model)
            change_points = detector.detect(data, method=method)

            # Save change points
            cp_data = [
                {
                    'date': cp.date.isoformat(),
                    'price_before': cp.price_before,
                    'price_after': cp.price_after,
                    'change_pct': cp.change_pct,
                    'direction': cp.direction,
                    'significance': cp.significance,
                    'confidence': cp.confidence
                }
                for cp in change_points
            ]
            save_results(cp_data, output_dir, 'change_points.json')

            # Generate change point plot
            visualizer.plot_price_with_change_points(
                data['Price'],
                change_points,
                save_path=output_dir / 'change_points.png'
            )

            logger.info(f"Detected {len(change_points)} change points")

            # Generate impact analysis
            if len(change_points) > 0:
                visualizer.plot_change_point_impact(
                    change_points,
                    save_path=output_dir / 'change_point_impact.png'
                )

        # Risk analysis
        risk_metrics = None
        if run_all or args.analyze_risk:
            logger.info("Running risk analysis...")

            risk_analyzer = RiskAnalyzer(config.risk)
            risk_metrics = risk_analyzer.calculate_metrics(data['Price'])

            # Save risk metrics
            risk_data = {
                'historical_volatility': risk_metrics.historical_volatility,
                'var_historical': risk_metrics.var_historical,
                'var_parametric': risk_metrics.var_parametric,
                'cvar': risk_metrics.cvar,
                'max_drawdown': risk_metrics.max_drawdown,
                'max_drawdown_duration': risk_metrics.max_drawdown_duration,
                'current_drawdown': risk_metrics.current_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'sortino_ratio': risk_metrics.sortino_ratio,
                'calmar_ratio': risk_metrics.calmar_ratio,
                'stress_test_results': risk_metrics.stress_test_results
            }
            save_results(risk_data, output_dir, 'risk_metrics.json')

            # Get risk summary
            risk_summary = risk_analyzer.get_risk_summary()
            save_results(risk_summary, output_dir, 'risk_summary.json')

            logger.info(f"Risk Summary: {risk_summary}")

        # Generate dashboard
        if run_all or args.dashboard:
            logger.info("Generating visualization dashboard...")

            # Ensure change points and risk metrics exist for dashboard
            if not change_points:
                detector = ChangePointDetector(config.model)
                change_points = detector.detect(data)

            if not risk_metrics:
                risk_analyzer = RiskAnalyzer(config.risk)
                risk_metrics = risk_analyzer.calculate_metrics(data['Price'])

            # Generate dashboard
            visualizer.plot_risk_dashboard(
                data['Price'],
                risk_metrics,
                change_points if change_points else None,
                save_path=output_dir / 'risk_dashboard.png'
            )

            logger.info(f"Dashboard saved to {output_dir / 'risk_dashboard.png'}")

        logger.info("Analysis complete!")
        logger.info(f"Results saved to {output_dir}")
        return 0

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
