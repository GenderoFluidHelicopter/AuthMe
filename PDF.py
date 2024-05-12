import os
import fitz
import shutil 

class PDF2JPG:
    def __init__(self):
        self.input_folder = './set'
        self.output_folder = './dataset'
        self.dpi = 300

    def convert_pdf_to_images(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        input_files = [f for f in os.listdir(self.input_folder) if f.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png'))]

        file_counter = 1

        for input_file in input_files:
            input_file_path = os.path.join(self.input_folder, input_file)
            
            if input_file.lower().endswith('.pdf'):
                pdf_document = fitz.open(input_file_path)

                for page_number in range(len(pdf_document)):
                    page = pdf_document.load_page(page_number)
                    image = page.get_pixmap(matrix=fitz.Matrix(self.dpi/72, self.dpi/72))
                    output_image_path = os.path.join(self.output_folder, f"{file_counter}.jpg")
                    file_counter += 1
                    image.save(output_image_path)

                pdf_document.close()
            else:
                output_image_path = os.path.join(self.output_folder, f"{file_counter}{os.path.splitext(input_file)[1]}") 
                file_counter += 1  
                shutil.copyfile(input_file_path, output_image_path)