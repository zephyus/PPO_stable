t1 = ('f3_5.23', 'f5_8.22', 'f5_8.20', 'f3_5.22', 'f3_5.21', 'f5_8.16', 'f5_8.15', 'f5_8.13', 'f5_8.11', 'f5_8.9')
t2 = ('f3_5.23', 'f5_8.22', 'f5_8.20', 'f3_5.22', 'f3_5.21', 'f5_8.16', 'f5_8.15', 'f5_8.13', 'f5_8.11')
t3 = ('f5_8.23', 'f3_5.23', 'f5_8.22', 'f5_8.20', 'f3_5.22', 'f3_5.21', 'f5_8.16', 'f5_8.15', 'f5_8.13', 'f5_8.11')
t4 = ('f5_8.23', 'f3_5.23', 'f5_8.22', 'f5_8.20', 'f3_5.22', 'f3_5.21', 'f5_8.16', 'f5_8.15', 'f5_8.13')
t5 = ()

def passed_cars(t1, t2):
    count = 0
    for t in t1[::-1]:
        if len(t2) == 0:
            return len(t1)
        elif t2[len(t2)-1] == t:
            break
        count += 1
    return count

if __name__=='__main__':
    print(passed_cars(t1, t2))
    print(passed_cars(t2, t3))
    print(passed_cars(t3, t4))
    print(passed_cars(t4, t5))
    print(passed_cars(t5, t1))

