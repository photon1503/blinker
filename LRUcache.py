from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size_bytes):
        self.cache = OrderedDict()
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0

    def get(self, key):
        if key in self.cache:
            # Move the accessed item to the end (most recently used)
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        # Calculate the size of the image in bytes
        value_size = value.nbytes
        
        # If the item is already in the cache, update the size
        if key in self.cache:
            self.current_size_bytes -= self.cache[key].nbytes
            self.cache.move_to_end(key)
        
        # Add the item to the cache
        self.cache[key] = value
        self.current_size_bytes += value_size
        
        # Evict items if the cache exceeds the max size
        while self.current_size_bytes > self.max_size_bytes:
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size_bytes -= oldest_value.nbytes

    def __contains__(self, key):
        return key in self.cache
