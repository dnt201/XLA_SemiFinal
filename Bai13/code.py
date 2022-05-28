import cv2
import numpy as np

if __name__ == '__main__':
    path = 'einstein.jpg'
    image = cv2.imread(path)
    rows,columns,channel = image.shape
    mask = np.ones((rows,columns),dtype=np.uint8) * 255
    r = 1
    c = 1

    
    color = (0,0,0)
    while(r < rows):
        while(c < columns and c < 319):
            # pix1_0 = image[c - 1 , r - 1,0]
            # pix1_1 = image[c - 1 , r - 1,1]
            # pix1_2 = image[c - 1 , r - 1,2]

            pix1_0 = image[r - 1 , c - 1,0]
            pix1_1 = image[r - 1 , c - 1,1]
            pix1_2 = image[r - 1 , c - 1,2]

            # pix2_0 = image[c , r - 1,0]
            # pix2_1 = image[c , r - 1,1]
            # pix2_2 = image[c , r - 1,2]

            pix2_0 = image[r -1, c,0]
            pix2_1 = image[r -1, c,1]
            pix2_2 = image[r -1, c,2]

            # pix3_0 = image[c + 1, r - 1,0]
            # pix3_1 = image[c + 1, r - 1,1]
            # pix3_2 = image[c + 1, r - 1,2]

            pix3_0 = image[r - 1, c + 1,0]
            pix3_1 = image[r - 1, c + 1,1]
            pix3_2 = image[r - 1, c + 1,2]

            # pix4_0 = image[c - 1, r,0]
            # pix4_1 = image[c - 1, r,1]
            # pix4_2 = image[c - 1, r,2]

            pix4_0 = image[r , c - 1,0]
            pix4_1 = image[r , c - 1,1]
            pix4_2 = image[r , c - 1,2]

            # pix5_0 = image[c, r,0]
            # pix5_1 = image[c, r,1]
            # pix5_2 = image[c, r,2]

            pix5_0 = image[r, c,0]
            pix5_1 = image[r, c,1]
            pix5_2 = image[r, c,2]

            # pix6_0 = image[c + 1, r,0]
            # pix6_1 = image[c + 1, r,1]
            # pix6_2 = image[c + 1, r,2]

            pix6_0 = image[r, c + 1,0]
            pix6_1 = image[r, c + 1,1]
            pix6_2 = image[r, c + 1,2]

            # pix7_0 = image[c - 1, r + 1,0]
            # pix7_1 = image[c - 1, r + 1,1]
            # pix7_2 = image[c - 1, r + 1,2]

            pix7_0 = image[r + 1, c - 1,0]
            pix7_1 = image[r + 1, c - 1,1]
            pix7_2 = image[r + 1, c - 1,2]

            # pix8_0 = image[c, r+1,0]
            # pix8_1 = image[c, r+1,1]
            # pix8_2 = image[c, r+1,2]

            pix8_0 = image[r+1, c,0]
            pix8_1 = image[r+1, c,1]
            pix8_2 = image[r+1, c,2]

            # pix9_0 = image[c + 1, r + 1,0]
            # pix9_1 = image[c + 1, r + 1,1]
            # pix9_2 = image[c + 1, r + 1,2]

            pix9_0 = image[r + 1, c + 1,0]
            pix9_1 = image[r + 1, c + 1,1]
            pix9_2 = image[r + 1, c + 1,2]

            # vertical line
            vertmean0 = (pix2_0 + pix5_0 + pix8_0) / 3
            vertmean1 = (pix2_1 + pix5_1 + pix8_1) / 3
            vertmean2 = (pix2_2 + pix5_2 + pix8_2) / 3
            vertvariance0 = (pow((pix2_0 - vertmean0), 2) + pow((pix5_0 - vertmean0), 2) + pow((pix8_0 - vertmean0), 2)) / 2
            vertvariance1 = (pow((pix2_1 - vertmean1), 2) + pow((pix5_1 - vertmean1), 2) + pow((pix8_1 - vertmean1), 2)) / 2
            vertvariance2 = (pow((pix2_2 - vertmean2), 2) + pow((pix5_2 - vertmean2), 2) + pow((pix8_2 - vertmean2), 2)) / 2
            vertvariance = vertvariance0 * vertvariance1 * vertvariance2

            # horizontal line
            horzmean0 = (pix4_0 + pix5_0+ pix6_0) / 3
            horzmean1 = (pix4_1 + pix5_1 + pix6_1) / 3
            horzmean2 = (pix4_2 + pix5_2 + pix6_2) / 3
            horzvariance0 = (pow((pix4_0 - horzmean0), 2) + pow((pix5_0 - horzmean0), 2) + pow((pix6_0 - horzmean0), 2)) / 2
            horzvariance1 = (pow((pix4_1 - horzmean1), 2) + pow((pix5_1 - horzmean1), 2) + pow((pix6_1 - horzmean1), 2)) / 2
            horzvariance2 = (pow((pix4_2 - horzmean2), 2) + pow((pix5_2 - horzmean2), 2) + pow((pix6_2 - horzmean2), 2)) / 2
            horzvariance = horzvariance0 * horzvariance1 * horzvariance2

            # downward diagonal line
            dndgmean0 = (pix1_0 + pix5_0 + pix9_0) / 3
            dndgmean1 = (pix1_1 + pix5_1 + pix9_1) / 3
            dndgmean2 = (pix1_2 + pix5_2 + pix9_2) / 3
            dndgvariance0 = (pow((pix1_0 - dndgmean0), 2) + pow((pix5_0 - dndgmean0), 2) + pow((pix9_0 - dndgmean0), 2)) / 2
            dndgvariance1 = (pow((pix1_1 - dndgmean1), 2) + pow((pix5_1 - dndgmean1), 2) + pow((pix9_1 - dndgmean1), 2)) / 2
            dndgvariance2 = (pow((pix1_2 - dndgmean2), 2) + pow((pix5_2 - dndgmean2), 2) + pow((pix9_2 - dndgmean2), 2)) / 2
            dndgvariance = dndgvariance0 * dndgvariance1 * dndgvariance2


            # upward diagonal line
            updgmean0 = (pix3_0 + pix5_0+ pix7_0) / 3
            updgmean1 = (pix3_1 + pix5_1 + pix7_1) / 3
            updgmean2 = (pix3_2 + pix5_2 + pix7_2) / 3
            updgvariance0 = (pow((pix3_0 - updgmean0), 2) + pow((pix5_0 - updgmean0), 2) + pow((pix7_0 - updgmean0), 2)) / 2
            updgvariance1 = (pow((pix3_1 - updgmean1), 2) + pow((pix5_1 - updgmean1), 2) + pow((pix7_1 - updgmean1), 2)) / 2
            updgvariance2 = (pow((pix3_2 - updgmean2), 2) + pow((pix5_2 - updgmean2), 2) + pow((pix7_2 - updgmean2), 2)) / 2
            updgvariance = updgvariance0 * updgvariance1 * updgvariance2


            # DRAW A LINE WHOSE LENGTH IS TO THE DEGREE THAT THE PIXEL IS IN THAT DIRECTION

            # whichever has the highest correlation, determine to what degree it has the best correlation
            if (vertvariance < horzvariance and vertvariance < dndgvariance and vertvariance < updgvariance):
                mask = cv2.line(mask, (c, r-1),(c,r+1), color)
            elif (horzvariance < vertvariance and horzvariance < dndgvariance and horzvariance < updgvariance):
                mask = cv2.line(mask, (c-1, r), (c + 1, r), color)
            elif (dndgvariance < vertvariance and dndgvariance < horzvariance and dndgvariance < updgvariance):
                mask = cv2.line(mask, (c-1, r-1), (c + 1, r+1), color)
            elif (updgvariance < vertvariance and updgvariance < horzvariance and updgvariance < dndgvariance):
                mask = cv2.line(mask, (c-1, r+1), (c + 1, r-1), color)
            else:
                mask = cv2.circle(mask,(c, r),0,color)
            c = c + 2
            if c == 317:
                c = 1
                break
        r = r + 2
        
    cv2.imshow("image",image)  
    cv2.imshow("mask",mask)  
    cv2.waitKey()