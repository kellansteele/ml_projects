# Kellan's Streamlit App for Machine Learning Projects
A streamlit app which allows you to explore some of my ML projects.

![Object Detection Page](/media/thermal_page.png)

In this app you will find information on image annotation (creating ground truths) for object detection datasets and some examples of image pre-processing steps using OpenCV.

In addition, there is an example of a YOLOv5 model I trained to detect people and dogs in thermal images.



# How to use Streamlit, OpenCV, and Heroku together:

Follow this ![tutorial](https://medium.com/analytics-vidhya/deploying-a-streamlit-and-opencv-based-web-application-to-heroku-456691d28c41) except the only change you need to make is to the setup.sh file. Instead of what is written in the tutorial, you should write:

```python
mkdir -p ~/.streamlit/

echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml```

Thanks to ![this comment](https://discuss.streamlit.io/t/deploying-heroku-error/1310/4) made by tim.

https://stackoverflow.com/questions/63845382/importerror-libgl-so-1-error-when-accessing-heroku-app