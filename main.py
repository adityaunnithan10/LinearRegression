import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')
# print(data.head(10))


# Gradient descent Function 
def gradient_descent(m_now,b_now,points,L) :
    m_gradient = 0
    b_gradient = 0

    n =  len(points) 
    for i in range(n) :
        x  = points.iloc[i].studytime
        y = points.iloc[i].score

        m_gradient += -(2/n) *(y-(m_now*x+b_now))*x
        b_gradient += -(2/n) *(y-(m_now*x+b_now))

    m = m_now - m_gradient*L
    b = b_now - b_gradient*L

    return m,b


def train(data,epochs=200000,L=0.00035) :
    m,b = 0,0
    for i in range(epochs) :
            m,b = gradient_descent(m,b,data,L) 
            if i % 10 == 0:
                print(f"epoch : {i}")
                print(f'M :{m}...B: {b}\n')
    return m,b


if __name__ == "__main__" :
    m,b = train(data)
    print(f'Final values : M :{m}....B :{b}')
    plt.scatter(data.studytime,data.score,color = "black")
    plt.plot(list(range(20,80)),[m*x+b for x in range(20,80)],color ="red")
    plt.show()