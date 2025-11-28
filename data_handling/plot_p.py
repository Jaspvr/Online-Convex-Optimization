import matplotlib.pyplot as plt

p_values = [-0.5,
            -0.55,
            -0.6,
            -0.65,
            -0.7,
            -0.75,
            -0.8,
            -0.85,
            -0.9,
            -0.95,
            -1.0]

wealth = [3.709381895478558,
          3.4070025232157595,
          2.895073278844201,
          2.766962485714421,
          2.6692142982731553,
          2.141526387741907,
          1.884575020374737,
          1.8559782487304934,
          1.6945661664624034,
          1.3341147403338023,
          1.1330270540194132]

p_values_rev = p_values[::-1]
wealth_rev = wealth[::-1]



plt.figure()
plt.plot(p_values_rev, wealth_rev, marker='o')
plt.title("Final Wealth vs $p$ (A$^p$ Preconditioner)")
plt.xlabel("$p$")
plt.ylabel("Final Wealth")
plt.grid(True)
plt.tight_layout()
plt.savefig("Plots/p05to1.pdf")  # vector graphic
plt.show()
