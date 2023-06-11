
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import pickle


def load_model():
    model = pickle.load(open('Digit_class.pickle', 'rb'))
    return model

st.write("""# Hand Digit Classification""")

SIZE = 192

canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150,width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    st.write('Write any digit between 0-9')
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pred = load_model().predict(test_x.reshape(1, -1))
    st.write(f'result: {pred}')
    # st.bar_chart(pred[0])