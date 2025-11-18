def count_words_in_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
            words = text.split()
            return len(words)
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# Example usage
if __name__ == "__main__":
    file_path = input("Enter the path to the text file: ")
    word_count = count_words_in_file(file_path)
    print(f"The number of words in the file is: {word_count}")