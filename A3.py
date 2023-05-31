import sys
import numpy as np
import random
from matplotlib import pyplot as plt
sys.setrecursionlimit(1000000)


## take depots,starting,endingposition,size of box

depots=[[0,0],[0,4],[4,0],[4,3]]
taxi_initial= [random.randrange(0,4,1), random.randrange(0,4,1)]
pas_initial=depots[random.choice([0,1,2,3])]  
A=depots[:]
A.remove(pas_initial)
dest_pos = A[random.choice([0,1,2])]
#taxi_intial=[3,3]
#pas_initial = [0,4]
#dest_pos = [4,3]

row_size = 5
col_size = 5

epsilon_A = 1e-10
epsilon = 0.1
alpha = 0.25


store_values_valueiteration = []
store_values_policyiteration = []


def rand(i):
    l=[0.05,0.05,0.05,0.05]
    l[i]=0.85
    list = np.random.choice([0,1,2,3],p=l,size=(100))
    return random.choice(list)

def plotV(L,title):
    x=[]
    y=[]
    for e in range(len(L)-1):
        x.append(e+1)
        y.append((max(abs(L[e][e1]-L[e+1][e1]) for e1 in range(len(L[0])))))
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('max_norm_distance')
    plt.plot(np.array(x),np.array(y))
    plt.savefig(title+'.jpg')
    plt.show()

def plotP(L,title):
    x=[]
    y=[]
    for e in range(len(L)-1):
        x.append(e+1)
        y.append((max(abs(L[e][e1]-L[len(L)-1][e1]) for e1 in range(len(L[0])))))
    plt.title(title)
    plt.xlabel('iterations')
    plt.ylabel('policy loss')
    plt.plot(np.array(x),np.array(y))
    plt.savefig(title+'.jpg')
    plt.show()


### create maze [L,T,R,B,D]
Maze = [[[0,0,1,1,1],[1,0,0,1,0],[0,0,1,1,0],[1,0,1,1,0],[1,0,0,1,1]],
        [[0,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0]],
        [[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0]],
        [[0,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,0,0,1],[0,1,1,0,0],[1,1,0,0,0],[0,1,1,0,1],[1,1,0,0,0]]]

MazE = [[[0,0,1,1,1],[1,0,1,1,0],[1,0,0,1,0],[0,0,1,1,0],[1,0,1,1,0],[1,0,1,1,1],[1,0,1,1,0],[1,0,0,1,0],[0,0,1,1,1],[1,0,0,1,0]],
        [[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,1],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,1],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0]],
        [[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0]],
        [[0,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,0,1,1],[0,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,1,1,0],[1,1,1,1,0],[1,1,0,1,0],[0,1,1,1,0],[1,1,0,1,0]],
        [[0,1,0,0,0],[0,1,1,0,0],[1,1,1,0,0],[1,1,0,0,0],[0,1,1,0,1],[1,1,1,0,0],[1,1,1,0,0],[1,1,0,0,0],[0,1,1,0,0],[1,1,0,0,1]]]


## define statespaces

def Statespace():
    S=[]
    for i in range(row_size):
        for j in range(col_size):
            taxi_pos = [i,j]
            S.append([taxi_pos,taxi_pos,1])
            for p in range(row_size):
                for q in range(col_size):
                    per_pos = [p,q]
                    S.append([taxi_pos,per_pos,0])
    return S

### for part A:

def pos_actions(s):
    A=[]
    t_pos=s[0]
    t_X = t_pos[0]
    t_Y = t_pos[1]
    L = Maze[t_X][t_Y]
    if(L[0]==1):
        A.append('left')
    if(L[1]==1):
        A.append('top')
    if(L[2]==1):
        A.append('right')
    if(L[3]==1):
        A.append('bottom')
    A.append('pickup')
    A.append('drop')
    return A

#### for  part B:

def poss_actions(s):
    A=[]
    A.append('left')
    A.append('top')
    A.append('right')
    A.append('bottom')
    A.append('pickup')
    A.append('drop')
    return A


####### MDP class

class MDP(object):
    def isEnd(self,state):
        return (state==[dest_pos,dest_pos,0])
    def actions(self,state):
        return pos_actions(state)
    def discount(self):
        return 0.99
    def succProbReward(self,state,action):
        result=[]
        navigations = {'right':[0,1],'left':[0,-1],'top':[-1,0],'bottom':[1,0]}
        t_X = state[0][0]
        t_Y = state[0][1]
        p_X = state[1][0]
        p_Y = state[1][1]
        pick = state[2]
        actions = pos_actions(state)
        if(action == 'pickup'):
            tax_pos = [t_X,t_Y]
            pas_pos = [p_X,p_Y]
            if(tax_pos==pas_pos):
                result.append(([tax_pos,pas_pos,1],1,-1))
            else:
                result.append(([tax_pos,pas_pos,0],1,-10))
        elif(action == 'drop'):
            tax_pos = [t_X,t_Y]
            pas_pos = [p_X,p_Y]
            if(tax_pos==pas_pos):
                if(tax_pos!=dest_pos):
                    result.append(([tax_pos,pas_pos,0],1,-1))
                else:
                    result.append(([tax_pos,pas_pos,0],1,20))
            else:
                result.append(([tax_pos,pas_pos,0],1,-10))
        else:
            for act in navigations:
                if(act in actions):
                    if(act==action):
                        tax_pos = [t_X+navigations[act][0],t_Y+navigations[act][1]]
                        pas_pos = [p_X,p_Y]
                        if(pick == 1):
                            pas_pos = tax_pos
                        result.append(([tax_pos,pas_pos,pick],0.85,-1))
                    else:
                        if((act!='pickup')&(act!='drop')):
                            tax_pos = [t_X+navigations[act][0],t_Y+navigations[act][1]]
                            pas_pos = [p_X,p_Y]
                            if(pick == 1):
                                pas_pos = tax_pos
                            result.append(([tax_pos,pas_pos,pick],0.05,-1))
                else:
                    if(act==action):
                        result.append((state,0.85,-1))
                    else:
                        result.append((state,0.05,-1))
        return result

    def states(self):
        return Statespace()

################################################################
################################################################
################################################################


### Model class

class Model:


    def __init__(self,pas_initial,dest_pos,statespace,maze):
        self.pas_ini = pas_initial
        self.des_pos = dest_pos
        self.endstate = [self.des_pos,self.des_pos,0]
        self.maze=maze
        self.stsp = statespace

    def set_taxi(self,initial_taxi_position):
        self.state = [initial_taxi_position,self.pas_ini,0]
    
    def do_action(self,action):    #state gets updated and reward is returned
        k=-1
        if(action=='top'):
            k=rand(1)
        elif (action=='bottom'):
            k=rand(3)
        elif(action == 'left'):
            k=rand(0)
        elif(action=='right'):
            k=rand(2)
        elif(action=='pickup'):
            if((self.state)[2]==0) and (self.state[1]==self.state[0]):
                self.state[2]=1
                return -1
            elif((self.state)[2]==0) and (self.state[1]!=self.state[0]):
                return -10
            elif(self.state[2]==1) and (self.state[1]!=self.state[0]):
                return -10
            else:
                return -1
        else:
            if(self.state[2]==1) and (self.state[0]==self.des_pos):
                self.state[2]=0
                return 20
            elif(self.state[2]==1):
                self.state[2]=0
                return -1
            elif(self.state[2]==0) and (self.state[0]!=self.state[1]):
                return -10
            else:
                return -1
        tX = self.state[0][0]
        tY = self.state[0][1]
        l=self.maze[tX][tY]
        if(k==0):
            if(l[0]==1):
                self.state[0][1] = tY-1
                if(self.state[2]==1):
                    self.state[1]=self.state[0]
                return -1
            else:
                return -1
        elif(k==1):
            if(l[1]==1):
                self.state[0][0] = tX-1
                if(self.state[2]==1):
                    self.state[1]=self.state[0]
                return -1
            else:
                return -1
        elif(k==2):
            if(l[2]==1):
                self.state[0][1] = tY+1
                if(self.state[2]==1):
                    self.state[1]=self.state[0]
                return -1
            else:
                return -1
        else:
            if(l[3]==1):
                self.state[0][0] = tX+1
                if(self.state[2]==1):
                    self.state[1]=self.state[0]
                return -1
            else:
                return -1


################################################################
################################################################
################################################################



def ValueIteration(mdp):
    V={}
    space = mdp.states()
    L1=[]
    for state in space:
        string = str(state[0])+" "+str(state[1])+" "+str(state[2])
        L1.append(0)
        V[string]=0
    store_values_valueiteration.append(L1)
    def Q(state,action):
        ans = 0
        for (newState,prob,reward) in mdp.succProbReward(state,action):
            string = str(newState[0])+" "+str(newState[1])+" "+str(newState[2])
            ans = ans+(prob*(reward+mdp.discount()*V[string]))
        return ans
    a=0
    while True:
        a=a+1;
        newV = {}
        for state in space:
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            if(mdp.isEnd(state)):
                newV[string]=0
            else:
                newV[string]= max(Q(state,'pickup'),Q(state,'drop'),Q(state,'right'),Q(state,'left'),Q(state,'top'),Q(state,'bottom'))
        if max(abs(V[string]-newV[string]) for string in V)<epsilon_A:
            break
        V=newV.copy()
        L=[]
        for st in V:
            L.append(V[st])
        store_values_valueiteration.append(L)

        ## optimal policy after each iteration
        pi = {}
        for state in mdp.states():
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            if mdp.isEnd(state):
                pi[string]='none'
            else:
                pi[string]= max((Q(state,action),action) for action in mdp.actions(state))[1]

        ### printing

        print(" ")
        print("Iteration "+str(a))
        print(" ")
        print('{:25} {:25} {:25}'.format('s','V(s)','pi(s)'))
        for state in mdp.states():
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            print('{:25} {:25} {:25}'.format(string,str(V[string]),str(pi[string])))
        #input()
    print("\ntotal iterations = "+str(a-1))

##################################################################
##################################################################
##################################################################


def PolicyIteration(mdp):
    Vpi = {}
    Pi ={}
    space = mdp.states()
    #L1=[]
    for state in space:
        string = str(state[0])+" "+str(state[1])+" "+str(state[2])
        Pi[string]='bottom'
        Vpi[string]=0
        #L1.append(0)
    #store_values_policyiteration.append(L1)
    ## intial policy is all bottom
    a=0
    while True:
        a=a+1
        newVpi,newPi= PolicyEvaluation(mdp,Pi,'Iterative')
        L=[]
        for st in Vpi:
            L.append(newVpi[st])
        store_values_policyiteration.append(L)
        if max(abs(Vpi[string]-newVpi[string]) for string in Vpi)<epsilon_A:
            break
        #b=0
        #for string in Vpi:
        #   if(newPi[string]==Pi[string]):
        #       b=b+1
        #if(b==len(space)):
        #   break
        Vpi=newVpi.copy()
        Pi =newPi.copy()

        #L=[]
        #for st in Vpi:
        #   L.append(Vpi[st])
        #store_values_policyiteration.append(L)


        ############# printing
        print(" ")
        print("Policy "+str(a))
        print(" ")
        print('{:25} {:25} {:25}'.format('s','V(s)','pi(s)'))
        for state in mdp.states():
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            print('{:25} {:25} {:25}'.format(string,str(Vpi[string]),str(Pi[string])))
    print("\n policies: "+str(a-1))


def PolicyEvaluation(mdp,Pi,strin):



    if(strin == 'oldlinear'):
        V={}
        space = mdp.states()
        for state in space:
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            V[string]=0

        def Q(state,action):
            ans = 0
            for (newState,prob,reward) in mdp.succProbReward(state,action):
                string = str(newState[0])+" "+str(newState[1])+" "+str(newState[2])
                ans = ans+(prob*(reward+mdp.discount()*V[string]))
            return ans

        while True:
            newV = V.copy()
            for state in space:
                string = str(state[0])+" "+str(state[1])+" "+str(state[2])
                if(mdp.isEnd(state)):
                    V[string]=0
                else:
                    V[string]=Q(state,Pi[string])
            if max(abs(V[string]-newV[string]) for string in V)<epsilon_A:
                break

        ## optimal policy after each iteration
        pi = {}
        for state in mdp.states():
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            if mdp.isEnd(state):
                pi[string]='none'
            else:
                pi[string]= max((Q(state,action),action) for action in mdp.actions(state))[1]
        return V,pi


    #iterative method
    if(strin  == 'Iterative'):
        V={}
        space = mdp.states()
        for state in space:
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            V[string]=0
        def Q(state,action):
            ans = 0
            for (newState,prob,reward) in mdp.succProbReward(state,action):
                string = str(newState[0])+" "+str(newState[1])+" "+str(newState[2])
                ans = ans+(prob*(reward+mdp.discount()*V[string]))
            return ans

        while True:
            newV = {}
            for state in space:
                string = str(state[0])+" "+str(state[1])+" "+str(state[2])
                if(mdp.isEnd(state)):
                    newV[string]=0
                else:
                    newV[string]=Q(state,Pi[string])
            if max(abs(V[string]-newV[string]) for string in V)<epsilon_A:
                break
            V=newV.copy()

        ## optimal policy after convergence
        pi = {}
        for state in mdp.states():
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            if mdp.isEnd(state):
                pi[string]='none'
            else:
                pi[string]= max((Q(state,action),action) for action in mdp.actions(state))[1]
        return V,pi


    if(strin == 'linear'):
        V={}
        space=mdp.states()
        m=len(space)
        A = np.zeros((m,m),dtype="float64")
        b = np.zeros(m,dtype="float64")
        #m=len(space)
        for i in range(0,m):
            A[i][i]=1
        for state in space:
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            ind = space.index(state)
            action = Pi[string]
            pos_states=mdp.succProbReward(state,action)
            for s1 in pos_states:
                newState,prob,reward = s1
                ind1 = space.index(newState)
                b[ind] += prob*reward
                A[ind][ind1] -= mdp.discount()*prob

        X=np.dot(np.linalg.inv(A),b.reshape(m,))
        fin_V = X.reshape(m).tolist()
        for i in range(0,m):
            state = space[i]
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            V[string]=fin_V[i]

        def Q(state,action):
            ans = 0
            for (newState,prob,reward) in mdp.succProbReward(state,action):
                string = str(newState[0])+" "+str(newState[1])+" "+str(newState[2])
                ans = ans+(prob*(reward+mdp.discount()*V[string]))
            return ans


        pi = {}
        for state in space:
            string = str(state[0])+" "+str(state[1])+" "+str(state[2])
            if mdp.isEnd(state):
                pi[string]='none'
            else:
                pi[string]= max((Q(state,action),action) for action in mdp.actions(state))[1]
        return V,pi


        



######################################################################
######################################################################
######################################################################


##part B

def Q_learning(n,alpha,epsilon,gamma,pas_position,dest_position,statespace,maze,decay):# n is no of episodes ,epsilon decays if decay ==1
    mdl = Model(pas_position,dest_position,statespace,maze)
    al = alpha
    d_reward_list = []
    Q = {}
    for state in statespace:
        for action in poss_actions(state):
            Q[(str(state),action)]=0
    for e in range(n):
        ep = epsilon
        dis_reward = 0
        i=0
        m = len(maze)
        ral=[]
        for j in range(m):
            ral.append(j)
        x = random.choice(ral)
        y = random.choice(ral)
        mdl.set_taxi([x,y])
        while(i<500 and (mdl.state!=mdl.endstate)):
            fac = gamma**i
            if(decay==1):
                ep = epsilon-(epsilon/555)*(i)
            #print(ep)
            g=np.random.rand()
            if(g<=ep):
                actions = poss_actions(mdl.state)
                action = random.choice(actions)
                prev_state = mdl.state
                r=mdl.do_action(action)
                Q[(str(prev_state),action)]=(1-al)*Q[(str(prev_state),action)]+(al)*(r+(gamma)*max(Q[(str(mdl.state),act)] for act in poss_actions(mdl.state)))
                dis_reward = dis_reward+fac*r
            else:
                re = max(Q[(str(mdl.state),act)] for act in poss_actions(mdl.state))
                b_act=""
                for act in poss_actions(mdl.state):
                    if(Q[(str(mdl.state),act)]==re):
                        b_act = act
                prev_state = mdl.state
                r=mdl.do_action(b_act)
                #print(prev_state)
                #print(b_act)
                #print(mdl.state)
                Q[(str(prev_state),b_act)] = (1-al)*Q[(str(prev_state),b_act)]+(al)*(r+(gamma)*max(Q[(str(mdl.state),action)] for action in poss_actions(mdl.state)))
                dis_reward = dis_reward+fac*r
            if(mdl.state==mdl.endstate):
                print(e)
            i=i+1
        d_reward_list.append([e,dis_reward])
    #print(d_reward_list)
    x_values=[]
    y_values=[]
    xp=0
    for i in range(n):
        if(i%10==0):
            xp=xp/10
            x_values.append(d_reward_list[i][0])
            y_values.append(xp)
            xp = 0
        else:
            xp=xp+d_reward_list[i][1]
    plt.title("Q-learning epsilon = 0.1 alpha = 0.5")
    plt.xlabel("no of episodes")
    plt.ylabel("sum of discounted rewards")
    x=np.array(x_values)
    y=np.array(y_values)
    plt.plot(x,y)
    plt.show()
    return 

def SARSA(n,alpha,epsilon,gamma,pas_position,dest_position,statespace,maze,decay):
    mdl=Model(pas_position,dest_position,statespace,maze)
    al = alpha
    d_reward_list = []
    Q={}
    for state in statespace:
        for action in poss_actions(state):
            Q[(str(state),action)]=0
    for e in range(n):
        ep =epsilon
        dis_reward = 0
        i=0
        m=len(maze)
        ral=[]
        for j in range(m):
            ral.append(j)
        x=random.choice(ral)
        y=random.choice(ral)
        mdl.set_taxi([x,y])
        while(i<500 and (mdl.state!=mdl.endstate)):
            fac = gamma**i
            if(decay==1):
                ep=epsilon-(epsilon/555)*(i)
            #print(ep)
            g=np.random.rand()
            prev_state = mdl.state
            r=0
            p_act=""
            if(g<=ep):
                actions = poss_actions(mdl.state)
                action = random.choice(actions)
                r=mdl.do_action(action)
                p_act = action
            else:
                re = max(Q[(str(mdl.state),act)] for act in poss_actions(mdl.state))
                b_act=""
                for act in poss_actions(mdl.state):
                    if(Q[(str(mdl.state),act)]==re):
                        b_act = act
                r=mdl.do_action(b_act)
                p_act = b_act
            #next_state = mdl.state
            dis_reward = dis_reward+fac*r
            g=np.random.rand()
            n_act=""
            if(g<=ep):
                actions = poss_actions(mdl.state)
                action = random.choice(actions)
                n_act = action
            else:
                re = max(Q[(str(mdl.state),act)] for act in poss_actions(mdl.state))
                for act in poss_actions(mdl.state):
                    if(Q[(str(mdl.state),act)]==re):
                        n_act = act
            Q[(str(prev_state),p_act)] = (1-al)*(Q[(str(prev_state),p_act)])+(al)*(r+(gamma)*(Q[(str(mdl.state),n_act)]))
            if(mdl.state == mdl.endstate):
                print(e)
            i=i+1
        d_reward_list.append([e,dis_reward])
    x_values=[]
    y_values=[]
    xp=0
    for i in range(n):
        if(i%10==0):
            xp=xp/10
            x_values.append(d_reward_list[i][0])
            y_values.append(xp)
            xp = 0
        else:
            xp=xp+d_reward_list[i][1]
    plt.title("SARSA ")
    plt.xlabel("no of episodes")
    plt.ylabel("sum of discounted rewards")
    x=np.array(x_values)
    y=np.array(y_values)
    plt.plot(x,y)
    plt.show()
    return

stsp = Statespace()

#######################################################################
########################################################################
######################################################################
mdp = MDP()

#ValueIteration(mdp)
#plotV(store_values_valueiteration,"valueiteration-0.99")
#PolicyIteration(mdp)
#plotP(store_values_policyiteration,"policyiteration(Iterative)-0.99")


####################################


"""
part b 1 - the code is already available
part b 2 - done with just changing decay parameters and functions
Q_learning(2000,alpha,epsilon,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,alpha,epsilon,gamma,pas_initial,dest_pos,stsp,Maze,1)
SARSA(2000,alpha,epsilon,gamma,pas_initial,dest_pos,stsp,Maze,0)
SARSA(2000,alpha,epsilon,gamma,pas_initial,dest_pos,stsp,Maze,1)
part b 3
SARSA(2000,alpha,epsilon,gamma,[0,0],dest_pos,stsp,Maze,1)
SARSA(2000,alpha,epsilon,gamma,[0,4],dest_pos,stsp,Maze,1)
SARSA(2000,alpha,epsilon,gamma,[4,0],dest_pos,stsp,Maze,1)
part b 4
varying epsilon alpha=0.1
Q_learning(2000,0.1,0,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.1,0.05,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.1,0.1,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.1,0.5,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.1,0.9,gamma,pas_initial,dest_pos,stsp,Maze,0)
varying alpha epsilon = 0.1
0.1  Q_learning(2000,0.1,0.1,gamma,pas_initial,dest_pos,stsp,Maze,0)
0.2  Q_learning(2000,0.2,0.1,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.3,0.1,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.4,0.1,gamma,pas_initial,dest_pos,stsp,Maze,0)
Q_learning(2000,0.5,0.1,gamma,pas_initial,dest_pos,stsp,Maze,0)
part b 5
row_size=10
col_size=10
stsp = Statespace()
dest_pos = [0,5]
SARSA(10000,alpha,epsilon,gamma,[0,0],dest_pos,stsp,MazE,1)
SARSA(10000,alpha,epsilon,gamma,[8,0],dest_pos,stsp,MazE,1)
SARSA(10000,alpha,epsilon,gamma,[3,3],dest_pos,stsp,MazE,1)
SARSA(10000,alpha,epsilon,gamma,[9,4],dest_pos,stsp,MazE,1)
SARSA(10000,alpha,epsilon,gamma,[0,8],dest_pos,stsp,MazE,1)
"""
