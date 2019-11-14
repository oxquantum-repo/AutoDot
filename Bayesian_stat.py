import numpy as np
from scipy.stats import invgamma

def calc_estimates(n, t):
    # posterior rate (lambda) is a Gamma distributiona (with Jeffreys prior of Poisson observations)
    a = n + 0.5 # 
    b = t

    p10 = invgamma.ppf(0.1, a=a, scale=b)
    p90 = invgamma.ppf(0.9, a=a, scale=b)
    mean = invgamma.mean(a=a, scale=b)
    return mean, p10, p90

#GPC2
time_gpc = (37559.818449974 + 41048.7140817642)/3600 # only for 500 iterations for each run
success_gpc = 7 + 6
print(calc_estimates(success_gpc, time_gpc))

time_us = 39127.290738821/3600 # only for 500 iterations
success_us = 0
print(calc_estimates(success_us, time_us))

time_ua = 2.0 * 123369.686498165/3600 # times two for another run
success_ua = 4
print(calc_estimates(success_ua, time_ua))

time_pr = 11.0 * 24801.4082360267/3600 # times 11 for another 10 runs
success_pr = 1
print(calc_estimates(success_pr, time_pr))

time_usp = 120432.01911211/3600
success_usp = 8
print(calc_estimates(success_usp, time_usp))

time_Basel2 = np.sum([8.238195230166117, 9.08703197936217, 8.208526365624534, 8.380222799314392, 7.686700075864792])
success_Basel2 = np.sum([6, 6, 5, 8, 4])
print(calc_estimates(success_Basel2, time_Basel2))
print(time_Basel2/success_Basel2)


print('-------------------------------------')
t1, s1 = 7.80321052160528, 3
t2, s2 = 7.719280686047342, 3
t3, s3 = 7.762990960346328, 2
t4, s4 = 7.712169336544143, 4
t5, s5 = 7.476775210698445, 0
t6, s6 = 8.079267378118303, 6

print(calc_estimates(s1+s2+s3, t1+t2+t3), (t1+t2+t3)/(s1+s2+s3))
print(calc_estimates(s4+s5+s6, t4+t5+t6), (t4+t5+t6)/(s4+s5+s6))
print(calc_estimates(s1+s2+s3+s4+s5+s6, t1+t2+t3+t4+t5+t6), (t1+t2+t3+t4+t5+t6)/(s1+s2+s3+s4+s5+s6))

