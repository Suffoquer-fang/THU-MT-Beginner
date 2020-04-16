file = open("japan",'r')
file2 = open('tmp', 'w')# 返回一个文件对象
list1= file.readlines()
list2 = []
j = -1
for i in list1:
    j+=1
    if j % 3 == 2:
        list1[j] = ("=== \n")
    else:
        list1[j] = i
    # print(j)
    # print(list1[j])
for i in list1:
    file2.write(i)




file.close()
file2.close()
file = open('tmp', 'r')
file2 = open("japan",'w')
file2.write(' '.join([f+ ' ' for fh in file for f in fh ]))
file.close()
file2.close()