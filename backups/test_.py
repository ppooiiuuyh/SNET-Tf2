min_step = 9999
def logic(n,  step=0):
    global min_step
    if step >= min_step:
        return

    if n == 1:
        if step < min_step :
            min_step = step
        return

    if n % 3 == 0 :
        logic(n//3, step+1)

    if n % 2 == 0 :
        logic(n//2, step+1)

    logic(n-1, step + 1 )

#n = int(input)
n = 10
logic(n)
print(min_step)