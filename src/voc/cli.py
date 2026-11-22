"""Command line: python -m voc.cli <stage>.

Stages, in order: download, build, replicate, test, global, report. Each writes its outputs to files, so a stage
can be rerun alone. `test` scores the post-sample months once: it refuses to run again unless --force (logged).
"""

import argparse
import logging
import sys

from voc import config


def setup_logging(stage: str) -> None:
    config.LOGS.mkdir(parents=True, exist_ok=True)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(config.LOGS / f"{stage}.log", encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", handlers=handlers, force=True)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="voc")
    p.add_argument("stage", choices=["download", "build", "replicate", "test", "global", "report"])
    p.add_argument("--force", action="store_true", help="test stage: allow a second (logged) post-sample run")
    p.add_argument("--seeds", type=int, default=None, help="override the number of RFF seeds (recorded in outputs)")
    a = p.parse_args(argv)
    setup_logging(a.stage)
    if a.stage == "download":
        from voc import download
        download.run()
    elif a.stage == "build":
        from voc import data
        data.build_all()
    elif a.stage == "replicate":
        from voc import pipeline
        pipeline.replicate(n_seeds=a.seeds)
    elif a.stage == "test":
        from voc import pipeline
        pipeline.test(force=a.force, n_seeds=a.seeds)
    elif a.stage == "global":
        from voc import pipeline
        pipeline.global_(n_seeds=a.seeds)
    elif a.stage == "report":
        from voc import report
        report.run()


if __name__ == "__main__":
    main()
