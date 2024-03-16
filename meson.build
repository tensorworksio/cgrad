project('cgrad', 'c', default_options : ['c_std=c99'])

cc = meson.get_compiler('c')

m_dep = cc.find_library('m', required : true)
criterion_dep = cc.find_library('criterion', required : true)
csptr_dep = cc.find_library('csptr', required : true)

deps = [m_dep, criterion_dep, csptr_dep]

inc_dir = include_directories('src')
src_dir = meson.current_source_dir() / 'src'
src_files = run_command('find', src_dir, '-name', '*.c', check : true).stdout().strip().split()

main_file = meson.current_source_dir() / 'main.c'
src_files += main_file

executable('main', src_files, dependencies : deps, include_directories : inc_dir)

test_dir = meson.current_source_dir() / 'tests'
test_files = run_command('find', test_dir, '-name', '*.c', check : true).stdout().strip().split()

foreach ts : test_files
  test_exe = executable(ts.underscorify(), src_files, dependencies : deps, include_directories : inc_dir)
  test(ts.underscorify(), test_exe)
endforeach