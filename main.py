from config import settings, setup_logging


def main() -> None:
    log = setup_logging()
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    log.info("%s started (env=%s)", settings.app_name, settings.env)


if __name__ == "__main__":
    main()
