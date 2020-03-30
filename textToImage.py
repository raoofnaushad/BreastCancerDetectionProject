import pytesseract       
  
# adds image processing capabilities 
from PIL import Image 

img = Image.open('1.png')      


result = pytesseract.image_to_string(img)    

with open('abc.txt',mode ='w') as file:      
      
                 file.write(result) 
                 print(result) 

