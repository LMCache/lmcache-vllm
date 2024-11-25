from setuptools import setup, find_packages

setup(
    name="lmcache_vllm",
    version="0.6.2.2",
    description = "lmcache_vllm: LMCache's wrapper for vllm",
    author = "LMCache team",
    author_email = "lmcacheteam@gmail.com",
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    packages=find_packages(),
    install_requires=[
        "lmcache==0.1.3",
        "vllm>=0.6.3.post1,<=0.6.3.post1",
    ],
    entry_points={
        'console_scripts': [
            "lmcache_vllm=lmcache_vllm.script:main"
        ],
    },
)

