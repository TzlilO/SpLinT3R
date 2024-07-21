import numpy as np
import torch
def parse_bpt(file_content, as_torch=True, device='cuda'):

    lines = file_content.strip().split('\n')
    num_patches = int(lines[0].strip())
    patches = []

    wrap = torch.tensor if as_torch else np.array

    index = 1
    degree_us, degree_vs = [], []
    for _ in range(num_patches):
        degree_u, degree_v = map(int, lines[index].strip().split())
        degree_us.append(degree_u), degree_vs.append(degree_v)
        control_points = []
        for i in range(4):  # Each patch has 4x4 control points
            for j in range(4):
                point = list(map(float, lines[index + 1].strip().split()))
                control_points.append(point)
                index += 1
        patches.append((degree_u, degree_v, wrap(control_points, device=device).reshape(4, 4, 3)))
        index += 1
    return patches


def read_bpt_file(file_path):
  with open(file_path, 'r') as f:
      return f.read()



def load_teapot():
  # Main execution
  file_path = 'experiments/teapot.txt'
  file_content = read_bpt_file(file_path)
  patches = parse_bpt(file_content)
  return patches

