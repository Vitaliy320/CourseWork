import graphviz


def parse_dot_file(path):
    test = graphviz.Source.from_file(path)
    v = 1
