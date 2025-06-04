import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io


def resize_and_pad_fixed(image, target_height, target_width):
    """
    Adds padding to the cropped and thresholded image to the exact target height and width (72dpi vs 96dpi, 72 being 126x160, 96 being 212x268) using white padding.
    The cropped image is centered.

    Args:
        image: The cropped and thresholded OpenCV image
        target_height: Final height of the output image
        target_width: Final width of the output image

    Returns:
        Padded image with the exact dimensions
    """
    h, w = image.shape[:2]  # dimensions of the cropped image

    # calculate padding for top, bottom, left, right to center the image
    top_padding = (target_height - h) // 2
    bottom_padding = target_height - h - top_padding
    left_padding = (target_width - w) // 2
    right_padding = target_width - w - left_padding

    # add padding with white color using cv
    padded_image = cv2.copyMakeBorder(
        image,
        top=top_padding,
        bottom=bottom_padding,
        left=left_padding,
        right=right_padding,
        borderType=cv2.BORDER_CONSTANT,
        value=255  # White padding
    )
    return padded_image


st.title("Crop, Threshold and Pad")

# upload an image
uploaded_file = st.file_uploader("Upload an Image (TIF, PNG, JPEG are allowed)", type=["tif", "png", "jpeg", "jpg"])

if uploaded_file is not None:
    # read
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    original_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        st.error("The uploaded file could not be read. Please upload a valid image.")
    else:
        # check DPI using pillow (72dpi vs 96dpi, 72 being 126x160, 96 being 212x268)
        pil_image = Image.open(uploaded_file)
        dpi = pil_image.info.get("dpi", (96, 96))[0]  # Default to 96 DPI if not present

        # handle special cases for DPI = 1 (metadata says 1 is dpi is 96)
        if dpi == 1:
            dpi = 96
        st.write(f"Detected DPI: {dpi} (assumed 96 DPI if unknown or invalid)")

        # fallback: If DPI is not 72, assume it's 96
        if dpi != 72:
            dpi = 96

        # display image
        st.image(original_image, caption="Original Image", use_container_width=True)

        # image dimensions
        image_height, image_width = original_image.shape

        # crop the image
        st.sidebar.header("Select Cropping Region")
        x_min = st.sidebar.slider("X Min:", 0, image_width - 1, 0)
        x_max = st.sidebar.slider("X Max:", 0, image_width - 1, image_width)
        y_min = st.sidebar.slider("Y Min:", 0, image_height - 1, 0)
        y_max = st.sidebar.slider("Y Max:", 0, image_height - 1, image_height)

        # ensure cropping dimensions
        if x_min >= x_max or y_min >= y_max:
            st.error("Invalid cropping dimensions. Ensure X Min < X Max and Y Min < Y Max.")
        else:
            cropped_image = original_image[y_min:y_max, x_min:x_max]
            st.image(cropped_image, caption="Cropped Region", use_container_width=True)

            # apply choosen thresholding
            st.sidebar.header("Thresholding Options")
            threshold_option = st.sidebar.radio("Select Thresholding Method:",
                                                ("Normal Otsu", "Gaussian-filtered Otsu"))

            if st.sidebar.button("Apply Thresholding"):
                if threshold_option == "Normal Otsu":
                    _, thresholded_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    st.image(thresholded_image, caption="Thresholded Image (Normal Otsu)", use_container_width=True)
                elif threshold_option == "Gaussian-filtered Otsu":
                    blurred_image = cv2.GaussianBlur(cropped_image, (5, 5), 0)
                    _, thresholded_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    st.image(thresholded_image, caption="Thresholded Image (Gaussian-filtered Otsu)",
                             use_container_width=True)

                # pad the image based on DPI (72dpi vs 96dpi, 72 being 126x160, 96 being 212x268)
                if dpi == 72:
                    final_image = resize_and_pad_fixed(thresholded_image, 160, 128)  # Height = 160, Width = 128
                elif dpi == 96:
                    final_image = resize_and_pad_fixed(thresholded_image, 268, 212)  # Height = 268, Width = 212

                # display the padded image
                st.image(final_image, caption="Final Padded Image", use_container_width=True)

                # save functionality using Streamlit download button ?
                is_success, buffer = cv2.imencode(".png", final_image)
                if is_success:
                    bytes_io = io.BytesIO(buffer)
                    st.download_button(
                        label="Download Final Image",
                        data=bytes_io.getvalue(),
                        file_name=f"{uploaded_file.name.split('.')[0]}_final.bmp",
                        mime="image/bmp"
                    )
                else:
                    st.error("Failed to save image. Please try again.")
