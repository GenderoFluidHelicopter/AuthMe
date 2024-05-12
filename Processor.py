import easyocr
import numpy as np
import os

class TextProcessor:
    def __init__(self):
        self.input_folder = './prepared'
        self.output_folder = './processed_text'
        self.reader = easyocr.Reader(['ru'])
        self.surname_dict = self.create_dictionary('./data/surnames.txt')
        self.name_dict = self.create_dictionary('./data/names.txt')
        self.patronymic_dict = self.create_dictionary('./data/patronymics.txt')

    def filter_text(self, results):
        cleaned_results = [''.join(filter(str.isalpha, text)) for text in results]
        filtered_results = [text.upper() for text in cleaned_results if text.isalpha() and len(text) > 3]
        while len(filtered_results) > 3:
            min_length = min(len(text) for text in filtered_results)
            filtered_results.remove(next(text for text in filtered_results if len(text) == min_length))
        
        return filtered_results

    def levenshtein_distance(self, s1, s2):
        m, n = len(s1), len(s2)
        dp = np.zeros((m+1, n+1))
        for i in range(m+1):
            dp[i][0] = i
        for j in range(n+1):
            dp[0][j] = j
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n]

    def find_best_match(self, word, dictionary):
        min_distance = float('inf')
        best_match = word
        for dict_word in dictionary:
            distance = self.levenshtein_distance(word, dict_word)
            if distance < min_distance:
                min_distance = distance
                best_match = dict_word
        return dictionary[best_match]

    def create_dictionary(self, file_path):
        dictionary = {}
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip().upper()
                dictionary[word] = word
        return dictionary

    def custom_spell_checker(self, words):
        corrected_words = []
        for i, word in enumerate(words):
            corrected_word = word
            if i == 0:
                corrected_word = self.find_best_match(word, self.surname_dict)
            elif i == 1:
                corrected_word = self.find_best_match(word, self.name_dict)
            elif i == 2:
                corrected_word = self.find_best_match(word, self.patronymic_dict)
            corrected_words.append(corrected_word)
        return corrected_words

    def process_text(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for file_name in os.listdir(self.input_folder):
            image_path = os.path.join(self.input_folder, file_name)
            
            result = self.reader.readtext(image_path, detail=0)
            print(f"Result from OCR for {file_name}:", result)
            
            filtered_result = self.filter_text(result)
            print("Filtered result:", filtered_result)
            
            if len(filtered_result) != 3:
                print("Unable to split text into surname, name, and patronymic for", file_name)
            else:
                corrected_words = self.custom_spell_checker(filtered_result)
                surname, name, patronymic = corrected_words
                
                output_file_path = os.path.join(self.output_folder, os.path.splitext(file_name)[0] + '.txt')

            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                output_file.write(f'{surname} {name} {patronymic}')

            print(f"Processed text saved to: {output_file_path}")