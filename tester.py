import numpy as np

rnew = np.array([[1,0],[1.1,1]])
rnew_diff = np.array([[0.1,0],[0.11,0],[0.001,0],[0.11,0],[0.17,0]])
print(rnew_diff[3:,0])
indexx = abs(rnew_diff[1: ,0] - 0.).argmin()
print(indexx)
# print(np.abs(rnew-rnew_diff))
# print(max(np.abs(rnew-rnew_diff)[0][0], np.abs(rnew-rnew_diff)[1][0] ))
# print(max.all(np.abs(rnew-rnew_diff), np.abs(rnew-rnew_diff) ))

# rel_error = max(abs((rnew-rsingstep)/rnew),
#                 abs((vnew-vsingstep)/vnew))


#t = [[]]
# t = []
# print(t)
# t.append([0,1])
# print(t)
# t.append([0,1])
# print(t)
# print(np.linalg.norm(t[0]))
# t = np.array([np.zeros(2)])
# print(t)
#
# t[0]= [0,1]
# print(t)
#
# rr = np.linalg.norm(t[:])
# print(rr)
#
# t = np.append(t,[[1,0]], axis = 0)
#
# print(t)
# rr = np.linalg.norm(t[:])
# print(rr)
#
# t = np.append(t,[[1,0]], axis = 0)
# rr = np.linalg.norm(t[:])
# print(rr)
