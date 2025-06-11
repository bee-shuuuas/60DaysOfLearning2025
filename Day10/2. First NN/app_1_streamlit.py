import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Global variables
dataIn = None
dataOut = None
testingIn = None
testingOut = None
trainingIn = None
trainingOut = None
weights = None
errorlist = []

# Functions
def generate_dataset(rangeData, lenData, testProportion):
    global dataIn, dataOut, testingIn, testingOut, trainingIn, trainingOut
    testEnd = round(lenData * testProportion)
    dataIn = np.random.randint(-rangeData, rangeData + 1, size=(int(lenData), 2))
    dataOut = dataIn[:, 0] + dataIn[:, 1]
    dataIn = np.concatenate([np.ones((lenData, 1)), dataIn], axis=1)
    testingIn = dataIn[0:testEnd]
    testingOut = dataOut[0:testEnd]
    trainingIn = dataIn[testEnd:]
    trainingOut = dataOut[testEnd:]
    return (
        f"Dataset generated! {lenData} samples.\n"
        f"Training samples: {len(trainingIn)}\n"
        f"Testing samples: {len(testingIn)}\n"
        f"Sample Training Input[0]: {trainingIn[0]}, Output: {trainingOut[0]}\n"
        f"Sample Testing Input[0]: {testingIn[0]}, Output: {testingOut[0]}"
    )

def initialize_weights(w0, w1, w2):
    global weights
    if w0 is None or w1 is None or w2 is None:
        weights = 4 * np.random.rand(3) - 2
    else:
        weights = np.array([w0, w1, w2])
    return f"Weights initialized: {weights}"

def calculateOut(x, w):
    return np.dot(x, w)

def gradient(x, w, correctValue):
    return 2 * (calculateOut(x, w) - correctValue) * x

def error(predictedValues, correctValues):
    return np.sum((predictedValues - correctValues) ** 2)

def accuracy(testingIn, testingOut, weights):
    return 1 - np.sum(np.abs(np.sign(np.round(testingOut - calculateOut(testingIn, weights))))) / len(testingOut)

def train_network(learningRate, iterations):
    global weights, errorlist
    if dataIn is None or weights is None:
        return "Please generate dataset and initialize weights first!", None, None
    errorlist = []
    lenTraining = len(trainingIn)
    errorlist.append(error(calculateOut(trainingIn[0], weights), trainingOut[0]))
    for i in range(iterations):
        index = np.random.randint(lenTraining)
        weights = weights - learningRate * gradient(trainingIn[index], weights, trainingOut[index])
        er = error(calculateOut(trainingIn[index], weights), trainingOut[index])
        errorlist.append(er)

    # Error Plot
    fig1, ax1 = plt.subplots()
    ax1.plot(range(iterations + 1), errorlist)
    ax1.set_title("Error vs Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Error")
    ax1.grid(True)

    # Weights Plot
    fig2, ax2 = plt.subplots()
    for i in range(3):
        ax2.plot([weights[i]] * len(errorlist), label=f'Weight[{i}]')
    ax2.set_title("Weights Evolution")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Weight Value")
    ax2.legend()
    ax2.grid(True)

    final_accuracy = accuracy(testingIn, testingOut, weights)
    return f"Training done! Final Accuracy: {final_accuracy*100:.2f}%", fig1, fig2

# ================= STREAMLIT UI ===================

st.set_page_config(page_title="Neural Network Trainer", layout="wide")
st.title("🧠 Manual Neural Network Trainer")

# Dataset generation
with st.expander("📊 Dataset Generator", expanded=True):
    rangeData = st.number_input("Range Data", value=20)
    lenData = st.number_input("Length of Dataset", value=1000)
    testProportion = st.slider("Test Proportion", min_value=0.0, max_value=1.0, value=0.3)

    if st.button("Generate Dataset"):
        info = generate_dataset(rangeData, lenData, testProportion)
        st.text_area("Dataset Info", value=info, height=160)

# Weight initialization
with st.expander("⚙️ Weight Initialization", expanded=True):
    w0 = st.number_input("Weight 0 (bias)", value=0.0)
    w1 = st.number_input("Weight 1", value=0.0)
    w2 = st.number_input("Weight 2", value=0.0)
    if st.button("Initialize Weights"):
        w_info = initialize_weights(w0, w1, w2)
        st.success(w_info)

# Training section
with st.expander("🛠️ Train Network", expanded=True):
    learningRate = st.number_input("Learning Rate", value=0.0001, format="%.6f")
    iterations = st.number_input("Iterations", value=10000, step=100)
    if st.button("Train Network"):
        output, fig1, fig2 = train_network(learningRate, int(iterations))
        st.success(output)
        if fig1: st.pyplot(fig1)
        if fig2: st.pyplot(fig2)

st.markdown("---")
st.caption("Made with  using Streamlit")
