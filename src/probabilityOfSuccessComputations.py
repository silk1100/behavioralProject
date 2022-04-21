def main():
    # User input
    # p = [108/1088, 133/1088, 320/1088, 97/1088]
    p = [1/6 for _ in range(10)]   
    # Algorithm implementation
    q = [1-pi for pi in p]
    p = [(f"p{i}", pr) for i, pr in enumerate(p)]
    q = [(f"q{i}", qr) for i, qr in enumerate(q)]
    symbSym = []
    propSum = []
    symb, probs = create_prob(p, q, 4,[], symbSym, propSum)
    
    
    # Print output
    # print(sum(propSum))
    print(symb)
    print(sum(propSum))
    x=0
    


def create_prob(p, q, k, included, symbSum=[], propSum=[]):
    if len(included) == k:
        mult = included.copy()
        prob = 1
        
        for i in included:
            prob *= p[i][1]

        for i in range(len(q)):
            if i in included:
                continue
            prob *= q[i][1]

        return sorted(mult), prob

    for i in range(len(p)):
        if i in included:
            continue
        included.append(i)

        symb, prob = create_prob(p, q, k, included, symbSum=symbSum, propSum=propSum)
        if symb not in symbSum:
            symbSum.append(symb)
            propSum.append(prob)

        included.pop()

    return symb, prob  


if __name__ == "__main__":
    main()