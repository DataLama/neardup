import redis
from omegaconf import OmegaConf

if __name__=='__main__':
    redis_config = {'host': '10.160.210.118', 'port': 6379}
    r = redis.Redis(**redis_config)
    r.flushdb()
