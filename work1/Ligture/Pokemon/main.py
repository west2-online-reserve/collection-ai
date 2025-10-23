import Pokemon
import shared
import Skills
import random

player_pokemons = []
ai_pokemons = []
Pokemon_list = [Pokemon.PikaChu,Pokemon.Bulbasaur,Pokemon.Charmander,Pokemon.Squirtle,Pokemon.Pidgey]
Pokemon_list_temp = Pokemon_list

counter_map = {
    shared.attr_water: [shared.attr_fire],
    shared.attr_fire: [shared.attr_grass],
    shared.attr_grass: [shared.attr_water, shared.attr_electric],
    shared.attr_electric: [shared.attr_water]

}

def set_battle_counters(pokemon1: Pokemon.Pokemon, pokemon2: Pokemon.Pokemon):
    # 先重置克制状态
    pokemon1.deactivate_counter()
    pokemon2.deactivate_counter()

    p1_attr = pokemon1.attribute
    p2_attr = pokemon2.attribute


    if p1_attr in counter_map and p2_attr in counter_map[p1_attr]:
        pokemon1.activate_counter()
        pokemon2.activate_be_countered()
        print(f"【属性克制】 -> 你的 {pokemon1.name} 克制对方的 {pokemon2.name}!")

    elif p2_attr in counter_map and p1_attr in counter_map[p2_attr]:
        pokemon2.activate_counter()
        pokemon1.activate_be_countered()
        print(f"【属性克制】 -> 对方的 {pokemon2.name} 克制你的 {pokemon1.name}!")

    else:
        print("【属性平和】 -> 双方属性没有克制关系。")

def pick_pokemon():
    print('\n请选择你的出战宝可梦:')
    for i, p in enumerate(player_pokemons):  # 来自chatgpt
        print(f'{i}.{p.name}')
    while True:
        try:
            first_index = int(input('输入编号:'))
            if first_index in range(len(player_pokemons)):
                shared.player_current_pokemon = player_pokemons[first_index]
                print(f'你的出战宝可梦为: {shared.player_current_pokemon.name}')
                break
            else:
                print('编号超出范围，请重新输入!')
        except ValueError:
            print('输入有误，请输入数字!')

print('请选择你的宝可梦:')
for i in range (len(Pokemon_list)):
    print(f'{i}.{Pokemon_list[i]().name}')
while True:
    try:
        player_pokemons = []
        choose = input('输入数字选择你的宝可梦(空格分隔):')
        index = choose.split(' ')
        for i in index:
            player_pokemons.append(Pokemon_list[int(i)]())
        if not len(player_pokemons) <=3 and len(player_pokemons) >0:
            print('只能选择1-3只宝可梦!')
            continue
        print('你选择了:',end='')
        for i in player_pokemons:
            print(i.name,end=' ')
        break
    except BaseException as e:
        print(f'输入有误，请重新输入!({e})')

for i in range(3):
    randpokemon = random.choice(Pokemon_list_temp)
    ai_pokemons.append(randpokemon())
    Pokemon_list_temp.remove(randpokemon)
print('电脑选择了:',end='')
for i in ai_pokemons:
    print(i.name, end=' ')
    i.belong = 'ai' #修改归属为ai

pick_pokemon() #选择出战宝可梦
# 电脑随机选择
ai_first_index = random.randint(0, len(ai_pokemons) - 1)
shared.ai_current_pokemon = ai_pokemons[ai_first_index]
print(f'电脑出战宝可梦为: {shared.ai_current_pokemon.name}')

current_turn = 1
while True:

    print(f'===================第{current_turn}回合===================')
    set_battle_counters(shared.player_current_pokemon, shared.ai_current_pokemon)
    shared.player_current_pokemon.turn_start()
    shared.ai_current_pokemon.turn_start()
    if not shared.player_current_pokemon.jump_turn:
        print(f'你的{shared.player_current_pokemon.name}的技能:')
        for i in range(len(shared.player_current_pokemon.skill)):
            print(f'{i}.{shared.player_current_pokemon.skill[i].name}')
        while True:
            try:
                index = int(input('输入数字选择技能:'))
                if index in range(len(shared.player_current_pokemon.skill)):
                    selected_skill: Skills.Skill = shared.player_current_pokemon.skill[index]
                    break
            except ValueError:
                print('输入有误，请重新输入!')
        selected_skill.use(shared.player_current_pokemon, shared.ai_current_pokemon)
        #print(f'<Debug>你的 {shared.player_current_pokemon.name} 当前数据:')
        #print(shared.player_current_pokemon)
        #print(f'<Debug>ai的 {shared.ai_current_pokemon.name} 当前数据:')
        #print(shared.ai_current_pokemon)
        print(f'你的 {shared.player_current_pokemon.name} 当前HP:{shared.player_current_pokemon.current_hp},对方的 {shared.ai_current_pokemon.name} 当前HP:{shared.ai_current_pokemon.current_hp}')
    else:
        print(f'你的{shared.player_current_pokemon.name}跳过了行动!')

    if shared.player_current_pokemon.current_hp <= 0:
        print(f'你的 {shared.player_current_pokemon.name} 昏厥!')
        player_pokemons.remove(shared.player_current_pokemon)
        if len(player_pokemons) == 0:
            print('你输了!')
            break
        pick_pokemon()
        continue

    if shared.ai_current_pokemon.current_hp <= 0:
        print(f'对方的 {shared.ai_current_pokemon.name} 昏厥!')
        ai_pokemons.remove(shared.ai_current_pokemon)
        if len(ai_pokemons) == 0:
            print('你赢了!')
            break
        else:
            ai_index = random.randint(0, len(ai_pokemons) - 1)
            shared.ai_current_pokemon = ai_pokemons[ai_index]
            print(f'电脑出战宝可梦为: {shared.ai_current_pokemon.name}')
            continue
    #ai随机使用一个技能
    if not shared.ai_current_pokemon.jump_turn:
        random.choice(shared.ai_current_pokemon.skill).use(shared.ai_current_pokemon, shared.player_current_pokemon)
        print(f'你的 {shared.player_current_pokemon.name} 当前HP:{shared.player_current_pokemon.current_hp},对方的 {shared.ai_current_pokemon.name} 当前HP:{shared.ai_current_pokemon.current_hp}')
    else:
        print(f'对方的{shared.ai_current_pokemon.name}跳过了行动!')
    # print(f'<Debug>你的 {shared.player_current_pokemon.name} 当前数据:')
    # print(shared.player_current_pokemon)
    # print(f'<Debug>ai的 {shared.ai_current_pokemon.name} 当前数据:')
    # print(shared.ai_current_pokemon)

    if shared.player_current_pokemon.current_hp <= 0:
        print(f'你的 {shared.player_current_pokemon.name} 昏厥!')
        player_pokemons.remove(shared.player_current_pokemon)
        if len(player_pokemons) == 0:
            print('你输了!')
            break
        pick_pokemon()
        continue
    if shared.ai_current_pokemon.current_hp <= 0:
        print(f'对方的 {shared.ai_current_pokemon.name} 昏厥!')
        ai_pokemons.remove(shared.ai_current_pokemon)
        if len(ai_pokemons) == 0:
            print('你赢了!')
            break
        else:
            ai_index = random.randint(0, len(ai_pokemons) - 1)
            shared.ai_current_pokemon = ai_pokemons[ai_index]
            print(f'电脑出战宝可梦为: {shared.ai_current_pokemon.name}')
            continue
    shared.player_current_pokemon.turn_end()
    shared.ai_current_pokemon.turn_end()
    current_turn += 1