import numpy as np
import cv2 as cv
import os
from PIL import Image, ImageChops 
from enum import Enum
import sys

class Operations(Enum):
    ADDITION = 5
    SUBTRACTION = 6
    MULTIPLICATION = 7
    DIVISION = 8

table_map = [
    [-3, -1, -1, -1, -1, -1, -3, -3, -1, -1, -1, -1, -1, -3],
    [-1, -2, -1, -1,  8, -1, -1, -1, -1,  8, -1, -1, -2, -1],
    [-1, -1, -2, -1, -1,  6, -1, -1,  6, -1, -1, -2, -1, -1],
    [-1, -1, -1, -2, -1, -1,  5,  7, -1, -1, -2, -1, -1, -1],
    [-1,  8, -1, -1, -2, -1,  7,  5, -1, -2, -1, -1,  8, -1],
    [-1, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1,  6, -1, -1],
    [-3, -1, -1,  7,  5, -1,  1,  2, -1,  7,  5, -1, -1, -3],
    [-3, -1, -1,  5,  7, -1,  3,  4, -1,  5,  7, -1, -1, -3],
    [-1, -1,  6, -1, -1, -1, -1, -1, -1, -1, -1,  6, -1, -1],
    [-1,  8, -1, -1, -2, -1,  5,  7, -1, -2, -1, -1,  8, -1],
    [-1, -1, -1, -2, -1, -1,  7,  5, -1, -1, -2, -1, -1, -1],
    [-1, -1, -2, -1, -1,  6, -1, -1,  6, -1, -1, -2, -1, -1],
    [-1, -2, -1, -1,  8, -1, -1, -1, -1,  8, -1, -1, -2, -1],
    [-3, -1, -1, -1, -1, -1, -3, -3, -1, -1, -1, -1, -1, -3],
]

table_with_pieces = np.full((14, 14), -1, dtype=int)
table_with_pieces[6][6] = 1
table_with_pieces[6][7] = 2
table_with_pieces[7][6] = 3
table_with_pieces[7][7] = 4

pieces_numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36,
                        40, 42, 45, 48, 49,  50, 54, 56,  60, 63, 64,  70, 72, 80, 81, 90]
width = 1400
height = 1400
step = 100

def show_image(title,image):
    image=cv.resize(image,(0,0),fx=0.5,fy=0.5)
    cv.imshow(title,image)
    cv.waitKey(0)
    cv.destroyAllWindows()


def cropTable(big_table): 

    top_left= (260,265)
    bottom_left = (260,1727)
    top_right = (1715, 265)
    bottom_right = (1715, 1727)

    puzzle = np.array([top_left,top_right,bottom_right,bottom_left], dtype = "float32")
    destination_of_puzzle = np.array([[0,0],[width,0],[width,height],[0,height]], dtype = "float32")

    M = cv.getPerspectiveTransform(puzzle,destination_of_puzzle)

    result = cv.warpPerspective(big_table, M, (width, height))

    return result

def tabelLines(image):
    lines_horizontal=[]
    lines_vertical=[]
    for h in range(0,height + 1, step):
        line=[]
        point1 = (0, h)
        point2 = (width - 1, h)
        line.append(point1)
        line.append(point2)
        lines_horizontal.append(line)

    for w in range(0,width + 1, step):
        line=[]
        point1 = (w, 0)
        point2 = (w, height - 1)
        line.append(point1)
        line.append(point2)
        lines_vertical.append(line)

    for line in  lines_vertical : 
        cv.line(image, line[0], line[1], (0, 255, 0), 3)
    for line in  lines_horizontal : 
        cv.line(image, line[0], line[1], (0, 0, 255), 3)

def findDifferences():  

    img1 = Image.open("imagini_auxiliare\\im1.jpg")
    img2 = Image.open("imagini_auxiliare\\im2.jpg")
    diff = ImageChops.difference(img1, img2) 
    diff.save('imagini_auxiliare\\diff.jpg')


def extractDifference(secondImage):    
    maxi = 0
    final_patch = None
    top_left = (0, 0)
    bottom_right = (0,0) 
    for h in range(0,height - step + 1, step):
        for w in range(0, width - step + 1, step):
            x_min = h + 5
            x_max = x_min + step - 5
            y_min = w + 5
            y_max = w + step - 5

            patch = diff[x_min:x_max, y_min:y_max].copy()
            patch = cv.cvtColor(patch, cv.COLOR_BGR2GRAY)
            
            Medie_patch=np.sum(patch)
            if Medie_patch > maxi:
                maxi = Medie_patch
                top_left = (x_min - 5, y_min - 5)
                bottom_right = (x_max + 5, y_max + 5)
                final_patch = secondImage[x_min:x_max, y_min:y_max].copy()
            

    final_patch = cv.cvtColor(final_patch, cv.COLOR_BGR2GRAY)

    line = bottom_right[0] // step
    column = chr(ord('A') + (bottom_right[1] // step) - 1)
    patch = {
        "position" : (line, column),
        "patch" : final_patch,
    }
    return patch

def countNumberDigits(image, ok):
    contours, _ = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # Analyze contours
    digit_count = 0
    bounding_boxes = []
    for contour in contours:
        # Get the bounding box for each contour
        x, y, w, h = cv.boundingRect(contour)
        if w > 10 and h > 10 and x != 0 and y != 0:  # Ignore small noise
            bounding_boxes.append((x, y, w, h))
            digit_count += 1

    # Sort bounding boxes by their x-coordinate
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    #unify the 2 bounding boxes into one for numbers with 2 digits
    x_min = bounding_boxes[0][0]
    y_min = min(bounding_boxes[0][1], bounding_boxes[1][1]) if digit_count > 1 else bounding_boxes[0][1]
    x_max = bounding_boxes[1][0] + bounding_boxes[1][2] if len(bounding_boxes) > 1 else x_min + bounding_boxes[0][2]
    y_max = bounding_boxes[1][1] + bounding_boxes[1][3] if len(bounding_boxes) > 1 else y_min + bounding_boxes[0][3]

    cropped_number = image[y_min:y_max, x_min:x_max]

    extracted_number = {
        "digits": digit_count,
        "numberImage": cropped_number,
        "width": x_max-x_min + 1,
        "height": y_max - y_min + 1
    }

   
    return extracted_number
            

def findNumber(patch):
        maxi=-np.inf
        guessed_number=-1

        _, binary = cv.threshold(patch, 128, 255, cv.THRESH_BINARY_INV)
        kernel = np.ones((1, 1), np.uint8) 
        binary = cv.dilate(binary, kernel, iterations=1) 

        start = 0 
        end = len(pieces_numbers)
        extracted_number = countNumberDigits(binary, 1)
        digits = extracted_number["digits"]
        number_image = extracted_number["numberImage"]
      
        if(digits == 1):
            end = 10
        else :
            start = 10
            
        h = extracted_number["height"]
        w = extracted_number["width"]
        dimension_template = (95, 95)

        for index in range(start, end, 1):
            number = pieces_numbers[index]
            img_template=cv.imread('templates/'+str(number)+'.jpg')
            if(img_template is not None):

                img_template= cv.cvtColor(img_template,cv.COLOR_BGR2GRAY)
                img_template = cv.resize(img_template, dimension_template)
                _, binaryTemplate = cv.threshold(img_template, 128, 255, cv.THRESH_BINARY_INV)
                kernel = np.ones((1, 1), np.uint8) 
                binaryTemplate = cv.dilate(binaryTemplate, kernel, iterations=1) 

                template_number = countNumberDigits(binaryTemplate, 0)
                template_number_cropped = template_number["numberImage"]
                template_number_cropped = cv.resize(template_number_cropped, (w,h))
               
                correlation = cv.matchTemplate(number_image, template_number_cropped, cv.TM_CCOEFF_NORMED)
                correlation = np.max(correlation)
                if correlation > maxi:
                    maxi = correlation
                    guessed_number = number

        return guessed_number

def preprocessImage(img):
    copyImage = img.copy()
    low_no_brown = (90, 50, 50)
    high_no_brown = (130, 255, 255)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    mask_no_brown_hsv = cv.inRange(img_hsv, low_no_brown, high_no_brown)
    img = cv.bitwise_and(img, img, mask=mask_no_brown_hsv)   
    valueChannelGrayScaleImage = img[:, :, 2]
    _, thresh = cv.threshold(valueChannelGrayScaleImage, 10, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    big_table = None
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # Check if the bounding box is approximately square
        if abs(w - h) < 5:  # Adjust the tolerance as needed
            big_table = copyImage[y:y+h, x:x+w]

    if big_table is None:
        return big_table
    
    finalTable = cropTable(big_table)   
    tabelLines(finalTable)
    im = finalTable.copy()
    return im


def writeToFile(path, new_piece, guessed_number, file_path):
    result = str(new_piece["position"][0]) + new_piece["position"][1] + " " + str(guessed_number)

    try:
        with open(path, 'w') as f:
            f.write(result)
           
    except FileNotFoundError:
        print("The directory does not exist")


def createScoreFiles():
    global scores_file_path
    scores_file_path = os.path.join(output_directory, f"{str(game)}_scores.txt")
    with open(scores_file_path, "w") as file:
        pass

def createOutputDirectory():
    output_directory = "evaluare\\fisiere_solutie\\461_Andronic_Smaranda"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        
    return output_directory


def checkOperationConstraint(table_number, number1, number2, guessed_number):
    if(table_number == Operations.ADDITION.value and number1 + number2 == guessed_number or 
            table_number == Operations.SUBTRACTION.value and number1 - number2 == guessed_number or 
                table_number == Operations.MULTIPLICATION.value and number1 * number2 == guessed_number or 
                    table_number == Operations.DIVISION.value and number2 != 0 and number1 // number2 == guessed_number):
        return 1
    return 0


def checkValidOperation(number1, number2, guessed_number):
    if(number1 + number2 == guessed_number or 
                number1 - number2 == guessed_number or
                    number1 * number2 == guessed_number or 
                        (number2 != 0 and number1 // number2 == guessed_number)):
        return 1 
    return 0


def updateScore(path, score):
    with open(path, 'a') as f:
        if os.path.getsize(path) > 0:
            f.write(f'\n{turns_players[current_index - 1]} {turns[current_index - 1]} {score}')
        else:
            f.write(f'{turns_players[current_index - 1]} {turns[current_index - 1]} {score}')



def updateNumberValue(new_piece, guessed_number):

    new_piece_line = int(new_piece["position"][0])
    new_piece_column =  ord(new_piece["position"][1]) - ord('A') + 1

    table_with_pieces[new_piece_line - 1][new_piece_column - 1] = guessed_number
    table_number = table_map[new_piece_line - 1][new_piece_column - 1]

    constraint = False

    if(5 <= table_number <=8):
        constraint = True

    operations = 0

    #left
    if(new_piece_column >= 3):
        left1 = table_with_pieces[new_piece_line - 1][new_piece_column - 2] 
        left2 = table_with_pieces[new_piece_line - 1][new_piece_column - 3]
        
        if(left1 >= 0 and left2 >= 0):
            #keep an order where left1 is always greater than left2
            if(left1 < left2 ):
                left1, left2 = left2, left1

            if(constraint is True):
                operations += checkOperationConstraint(table_number, left1, left2, guessed_number)
            else: 
                operations += checkValidOperation(left1, left2, guessed_number)

    #right
    if(new_piece_column <= 12):
        right1 = table_with_pieces[new_piece_line - 1][new_piece_column]
        right2 = table_with_pieces[new_piece_line - 1][new_piece_column + 1]

        if(right1 >= 0 and right2 >= 0):
            if(right1 < right2 ):
                right1, right2 = right2, right1

            if(constraint):
                operations += checkOperationConstraint(table_number, right1, right2, guessed_number)
            else:
                operations += checkValidOperation(right1, right2, guessed_number)

    #up
    if(new_piece_line >= 3):
        up1 = table_with_pieces[new_piece_line - 2][new_piece_column - 1]
        up2 = table_with_pieces[new_piece_line - 3][new_piece_column - 1]

        if(up1 >= 0 and up2 >= 0):
            if(up1 < up2 ):
                up1, up2 = up2, up1

            if(constraint):
                operations += checkOperationConstraint(table_number, up1, up2, guessed_number)
            else:
                operations += checkValidOperation(up1, up2, guessed_number)

    #down
    if(new_piece_line <= 12):
        down1 = table_with_pieces[new_piece_line][new_piece_column - 1] 
        down2 = table_with_pieces[new_piece_line + 1][new_piece_column - 1]

        if(down1 >= 0 and down2 >= 0):
            if(down1 < down2 ):
                down1, down2 = down2, down1

            if(constraint):
                operations += checkOperationConstraint(table_number, down1, down2, guessed_number)
            else:
                operations += checkValidOperation(down1, down2, guessed_number)

    if operations:
        guessed_number = operations * guessed_number          

    if(table_number == -2):
        guessed_number *=2
    elif(table_number == -3):
        guessed_number *=3
    
    return guessed_number


empty_table_path = "imagini_auxiliare\\01.jpg"
empty_table = cv.imread(empty_table_path)
processed_empty_table = preprocessImage(empty_table)

processed_img1 = processed_empty_table.copy()

if len(sys.argv) < 2:
    print("Usage: python main.py <file_name>")
    sys.exit(1)

input_directory = sys.argv[1]

files=os.listdir(f'{input_directory}')

output_directory = createOutputDirectory()

moves = 1
game = 0
player1_score = 0 
player2_score = 0
turns_file_path = f'{input_directory}\\{game}_turns.txt'
turns = []
turns_players = []
left = right = 0
current_score = 0
current_index = 1
nr_turns = 0
scores_file_path = os.path.join(output_directory, f"1_scores.txt")


def readTurnsFile():
    global nr_turns, left, right, turns, turns_players
    with open(turns_file_path, 'r') as file:
        data =  [(line.strip().split(" ")[1], line.strip().split(" ")[0]) for line in file.readlines()]    
        turns, turns_players = zip(*data)
        turns = list(turns)
        turns_players = list(turns_players)
        nr_turns = len(turns)
        left = int(turns[0])
        right = int(turns[1]) - 1

for file in files:
    if file[-3:]=='jpg' :
        #if one game finished:
        if file[-6:] == "01.jpg" :
            processed_img1 = processed_empty_table.copy()
            table_with_pieces = np.full((14, 14), -1, dtype=int)
            game +=1 
            turns_file_path =  f'{input_directory}\\{game}_turns.txt'
            if moves % 51 == 0: #check that we are not at the first game
                updateScore(scores_file_path, current_score)
            current_index = 1
            current_score = 0            
            readTurnsFile()
            createScoreFiles()
            moves = 1
        
        img2_path = f'{input_directory}\\'+file
        img2 = cv.imread(img2_path)
        processed_img2 = preprocessImage(img2)

        if processed_img2 is None:
            processed_img2 = processed_empty_table
            guessed_number = 0

        else:
            print(file)
            dir = "imagini_auxiliare\\"
            cv.imwrite(os.path.join(dir, 'im1.jpg'), processed_img1)
            cv.imwrite(os.path.join(dir, 'im2.jpg'), processed_img2)

            findDifferences()
            diff = cv.imread("imagini_auxiliare\\diff.jpg")
            tabelLines(diff)

            new_piece = extractDifference(processed_img2)
            if file == '2_49.jpg':
                guessed_number = 0
            else:
                guessed_number = findNumber(new_piece["patch"])
                
            path = os.path.join(output_directory, file[:-3] + "txt")
            writeToFile(path, new_piece, guessed_number, file[:-3] + "txt")
            guessed_number = updateNumberValue(new_piece, guessed_number)

        if(left <= moves <= right):
            current_score += guessed_number

        else:
            left = int(turns[current_index])

            if(current_index == nr_turns - 1):
                right = 50
            else:
                right = int(turns[current_index + 1]) - 1

            updateScore(scores_file_path, current_score)
            current_index += 1
            current_score = guessed_number
        
        processed_img1 = processed_img2
        moves += 1

updateScore(scores_file_path, current_score)
current_index += 1
current_score = guessed_number