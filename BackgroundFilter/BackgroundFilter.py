import cv2
import numpy as np

video = cv2.VideoCapture(0)
background_image = cv2.imread("download.jpg")


background_image = cv2.resize(background_image, (640, 480))  # Adjust the size as needed


while True:

    ret, frame = video.read()
    mask = cv2.createBackgroundSubtractorMOG2().apply(frame)
    
    # Invert the mask to get the foreground
    foreground = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Resize the mask to match the background image
    mask = cv2.resize(mask, (background_image.shape[1], background_image.shape[0]))
    
    # Convert the mask to the appropriate data type
    mask = np.uint8(mask)
    
    # Apply the inverse mask to the background image
    background = cv2.bitwise_and(background_image, background_image, mask=cv2.bitwise_not(mask))
    
    # Combine the foreground and background
    result = cv2.add(foreground, background)
    
    # Display the result
    cv2.imshow("Background Filter", result)
    
    # Check for the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video.release()
cv2.destroyAllWindows()




# import cv2
# import numpy as np

# # Load the video and background image
# video = cv2.VideoCapture(0)  # 0 represents the default camera
# background_image = cv2.imread("download.jpg")

# # Read the video frame by frame
# while True:
#     # Capture the current frame
#     ret, frame = video.read()
    
#     # Convert the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
#     # Apply adaptive thresholding to separate the foreground (face) from the background
#     mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
#     # Resize the mask to match the frame size
#     mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
#     # Convert the mask to the appropriate data type and ensure it has the same size as the frame
#     mask = np.uint8(mask)
#     mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    
#     # Invert the mask to get the foreground
#     mask_inv = cv2.bitwise_not(mask)
    
#     # Apply the mask to the frame and the background image
#     foreground = cv2.bitwise_and(frame, mask_inv)
#     background = cv2.bitwise_and(background_image, mask)
    
#     # Combine the foreground and background
#     result = cv2.add(foreground, background)
    
#     # Display the result
#     cv2.imshow("Background Filter", result)
    
#     # Check for the 'q' key to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the video capture and close the windows
# video.release()
# cv2.destroyAllWindows()



# # import cv2
# # import cvzone
# # from cvzone.SelfiSegmentationModule import SelfiSegmentation
# # import os

# # cap = cv2.VideoCapture(0)
# # cap.set(3, 640)
# # cap.set(4, 480)
# # # cap.set(cv2.CAP_PROP_FPS, 60)

# # segmentor = SelfiSegmentation()
# # fpsReader = cvzone.FPS()

# # # imgBG = cv2.imread("BackgroundImages/3.jpg")

# # listImg = os.listdir("BackgroundImages")
# # imgList = []
# # for imgPath in listImg:
# #     img = cv2.imread(f'BackgroundImages/{imgPath}')
# #     imgList.append(img)

# # indexImg = 0

# # while True:
# #     success, img = cap.read()
# #     # imgOut = segmentor.removeBG(img, (255,0,255), threshold=0.83)
# #     imgOut = segmentor.removeBG(img, imgList[indexImg], threshold=0.8)

# #     imgStack = cvzone.stackImages([img, imgOut], 2,1)
# #     _, imgStack = fpsReader.update(imgStack)
# #     print(indexImg)
# #     cv2.imshow("image", imgStack)
# #     key = cv2.waitKey(1)
# #     if key == ord('a'):
# #         if indexImg>0:
# #             indexImg -=1
# #     elif key == ord('d'):
# #         if indexImg<len(imgList)-1:
# #             indexImg +=1
# #     elif key == ord('q'):
# #         break
