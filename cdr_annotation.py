import re
W_RPattern = re.compile(r'W\wR')
Y_CPattern = re.compile(r'(Y\wC)|(YY\w)')
WG_GPattern = re.compile(r'(WG\wG)|(W\wQG)|(\wGQG)|(\w\wQG)|(WG\w\w)|(W\w\wG)|(W\wQ\w)')
def find_cdr1(sequence):
    # STARTING POS. OF CDR1:
    left_area = sequence[20:26]  # look from pos. 20 - 26 of seq (0-based)
    left_cdr = -1
    la_i = left_area.find('SC')
    if la_i < 0:
        # didn't find 'SC', look for 'C'
        la_i = left_area.find('C')
    else:
        la_i += 1  # 'C' is our marker, so advance past 'S'
    if la_i >= 0:
        left_cdr = la_i + 20 + 5  # CDR1 starts at 'C' + 5 (add 20 to put it back in the full sequence)

    # ENDING POS. OF CDR1:
    right_area = sequence[32:40]  # look from pos. 32 - 40 of seq (0-based)
    ra_i = -1
    right_cdr = -1
    W_R = W_RPattern.search(right_area)
    if W_R != None:
        # if we found 'WXR', find its index
        ra_i = right_area.find(W_R[0])

    else:
        ra_i = right_area.find('W')  # didn't find 'WXR', look for 'W'
    if ra_i >= 0:
        right_cdr = ra_i + 32 - 1 + 1  # CDR1 ends at 'W' - 1 (add 32 to put it back in the full sequence)

    # check if st/end found and if not follow rules:
    if left_cdr == -1 and right_cdr == -1:
        left_cdr = 28
        right_cdr = 36
    elif left_cdr == -1:
        left_cdr = right_cdr - 8
    elif right_cdr == -1:
        right_cdr = left_cdr + 8

    return [left_cdr,right_cdr]

def find_cdr2(sequence):
    # STARTING POS. OF CDR2:
    left_area = sequence[32:40]  # look from pos. 32 - 40 of seq (0-based)
    la_i = -1
    left_cdr = -1
    W_R = W_RPattern.search(left_area)
    if W_R != None:
        # if we found 'WXR', find its index
        la_i = W_R.start(0)
    else:
        la_i = left_area.find('W')  # didn't find 'WXR', look for 'W'
    if la_i >= 0:
        left_cdr = la_i + 32 + 14  # CDR2 starts at 'W' + 14 (add 32 to put it back in the full sequence)

    # ENDING POS. OF CDR2:
    right_area = sequence[63:72]  # look from pos. 63 - 72 of seq (0-based)
    right_cdr = -1
    ra_i = right_area.find('RF')
    if ra_i >= 0:
        right_cdr = ra_i + 63 - 8 + 1  # CDR2 ends at 'R' - 8 (add 63 to put it back in the full sequence)

    # check if st/end found and if not follow rules:
    if left_cdr == -1 and right_cdr == -1:
        left_cdr = 51
        right_cdr = 60
    elif left_cdr == -1:
        left_cdr = right_cdr - 9
    elif right_cdr == -1:
        right_cdr = left_cdr + 9

    return [left_cdr,right_cdr]

def find_cdr3(sequence):
    left_area = sequence[90:105]
    la_i = -1
    left_cdr = -1
    Y_C = Y_CPattern.search(left_area)
    if Y_C != None:
        # if we found 'YXR', find its index
        la_i = Y_C.start(0)+2
    else:
        la_i = left_area.find('C')  # didn't find 'YXC', look for 'C'

    if la_i >= 0:
        left_cdr = la_i + 90 + 3
    n = len(sequence) - 1
    n1 = n - 14
    subtract_amount = 1
    right_area = sequence[n1:n - 4]
    ra_i = -1
    right_cdr = -1
    WG_G = WG_GPattern.search(right_area)
    if WG_G != None:
        ra_i = WG_G.start(0)

    if ra_i >= 0:
        right_cdr = ra_i + n1 - subtract_amount + 1  # CDR3 ends at 'W' - 1 (or 'Q' - 3) (add n-14 to put it back in the full sequence)
    # check
    if left_cdr == -1 and right_cdr == -1:
        left_cdr = n - 21
        right_cdr = n - 10
    elif left_cdr == -1:
        left_cdr = right_cdr - 11
    elif right_cdr == -1:
        if left_cdr + 11 <= n:
            right_cdr = left_cdr + 11
        else:
            right_cdr = n
    if left_cdr > right_cdr:
        left_cdr = n - 1
        right_cdr = n
    return [left_cdr,right_cdr]