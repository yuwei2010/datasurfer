import matplotlib.pyplot as plt

# Test case 1: Basic usage
ax = plt.subplot()
text = "Hello world! This is a test."
plot_wordcloud(ax, text)
plt.show()

# Test case 2: Customizing word cloud appearance
ax = plt.subplot()
text = "Hello world! This is a test."
kwargs = {'background_color': 'white', 'colormap': 'cool', 'height': 800, 'width': 1200}
plot_wordcloud(ax, text, **kwargs)
plt.show()

# Test case 3: Empty text
ax = plt.subplot()
text = ""
plot_wordcloud(ax, text)
plt.show()