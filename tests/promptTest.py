# Read query from prompt.txt file
    try:
        with open('tests/prompt.txt', 'r') as file:
            query = file.read().strip()
        if not query:
            logger.error("tests/prompt.txt file is empty")
            print("Error: tests/prompt.txt file is empty")
            sys.exit(1)
        logger.info(f"Read query from tests/prompt.txt: {query}")
    except FileNotFoundError:
        logger.error("tests/prompt.txt file not found")
        print("Error: tests/prompt.txt file not found")
        sys.exit(1)