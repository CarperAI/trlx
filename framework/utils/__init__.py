def chunk(L, chunk_size):
    return [L[i:i+chunk_size] for i in range(0, len(L), chunk_size)]