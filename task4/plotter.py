import matplotlib.pyplot as plt 

if __name__ == "__main__":

    # interval = 50
    # with open('LDC_result/log', 'r') as f:
    #     data = f.readlines()

    #     x = []
    #     y = []
    #     for i, line in enumerate(data):
    #         if i % interval != 0: continue
    #         step = int(line.split(', ')[1].split(': ')[1])
    #         loss = float(line.split(', ')[2].split(': ')[1])
    #         x.append(step)
    #         y.append(loss)

    # plt.plot(x, y, label='en')

    # with open('LDC-zh_result/log', 'r') as f:
    #     data = f.readlines()

    #     x = []
    #     y = []
    #     for i, line in enumerate(data):
    #         if i % interval != 0: continue
    #         step = int(line.split(', ')[1].split(': ')[1])
    #         loss = float(line.split(', ')[2].split(': ')[1])
    #         x.append(step)
    #         y.append(loss)

    # plt.plot(x, y, label='zh')
    # plt.xlabel('global steps')
    # plt.ylabel('loss')
    # plt.legend()
    # plt.show()



    with open('nohup.log', 'r') as f:   
        data = f.readlines()
        x = []
        y = []
        for line in data:
            if 'step = ' in line and 'loss = ' in line:
                loss = line.split('loss = ')[1].split(',')[0]
                loss = float(loss)
                y.append(loss)
    plt.plot(y, label='loss')
    plt.xlabel('global steps')
    plt.ylabel('loss')
    plt.legend()
    plt.show()
        